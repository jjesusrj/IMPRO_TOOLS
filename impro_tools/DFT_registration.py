# ----------------------------------------------------------------------------
#
# PyTorch-based efficient subpixel image registration by crosscorrelation. 
#
#           This is a PyTorch implemetation of the DFT registration based on MATLAB version posted in: 
#           https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
#
#           Purpose:
#               1) Estimate subpixel shifts between two images.
#               2) Register the moving image to the fixed image using the estimated shifts. 
#               3) speedup the computations using PyTorch and GPU acceleration.
#               4) Easy integration to C++ though LibTorch.
#
#           Usage:
#               Create an instance of the DFT registration class with the size of the image, the upsampling factor (usfac):
#               1) Define the DFT registrator:
#                   Numpy version:
#                       dft_reg = Numpy_DFT_Registrator(fixed_img.shape, usfac=20)
#                   PyTorch version:
#                       dft_reg = Torch_DFT_Registrator(fixed_img.shape, usfac=20, device="cuda:0")
#
#               2) Perform the registration:
#               shifts, registered = dft_reg.execute(fixed_img, moving_img)
#
#
# ----------------------------------------------------------------------------
# When using this code, please cite:
#   Jose J. Rico-Jimenez, et al., "Real-time OCT image denoising using a self-fusion neural network," 
#   Biomed. Opt. Express 13, 1398-1409 (2022)
#
# ----------------------------------------------------------------------------
#
# Created by Jose Rico - March 2021
# 
# Diagnostic Imaging and Image-Guided Interventions (DIIGI) LAB
# Vanderbilt University
#
# ----------------------------------------------------------------------------




import numpy as np
import math 
import torch


# Numpy-based DFT-based subpixel image registration class
class Numpy_DFT_Registrator():
    '''
    DFT-based subpixel shift estimation and Image registration.\n
    '''

    def __init__(self, img_size, usfac=20):
        '''DFT-based subpixel shift estimation and Image registration.\n
        ----------------------------------------------------
            `Parameters`:\n
                img_size: [Rows, Cols] Number of rows and columns of the image.\n
                usfac: Upsampling factor. The image shifts and registeration \
                       will be computed to 1/usfac fraction of a pixel.\n  
        '''
        nr, nc = img_size
        self.usfac = usfac
        self.pi = math.pi
        self.cx_2pi = 1j * 2 * self.pi 
        
        img_size = img_size
        self.nr  = img_size[0]
        self.nc  = img_size[1]
        self.FTpad_outsize = [2*self.nr, 2*self.nc]
        
        self.total_num_pixels =  nr * nc
        self.dftshift  = np.fix(np.ceil(self.usfac*1.5)/2)

        # Constant parameters for the function "dftups"
        self.nor = np.ceil(usfac*1.5)
        self.noc = np.ceil(usfac*1.5)
        self.const_nc = ( -self.cx_2pi / (self.nc*self.usfac) ) * \
                    ( np.expand_dims(np.fft.ifftshift(np.arange(self.nc) - np.floor(self.nc/2) ), axis=1) )
        self.const_nr = ( -self.cx_2pi / (self.nr*self.usfac) ) * \
                    ( np.fft.ifftshift(np.arange(self.nr)) - np.floor(self.nr/2) ) 

        self.const_range_nc = np.arange(self.noc)
        self.const_range_nr = np.arange(self.nor)
        
        self.Nr  = np.fft.ifftshift( np.arange(-int(nr/2), int(np.ceil(nr/2)),dtype=int) )
        self.Nc  = np.fft.ifftshift( np.arange(-int(nc/2), int(np.ceil(nc/2)),dtype=int) )
        self.Nr2 = np.fft.ifftshift( np.arange(int(-np.fix(nr)), int(np.ceil(nr)),dtype=int) )
        self.Nc2 = np.fft.ifftshift( np.arange(int(-np.fix(nc)), int(np.ceil(nc)),dtype=int) )
        [self.Ncf, self.Nrf] = np.meshgrid(self.Nc, self.Nr, indexing='ij')
        [self.Ncf, self.Nrf] = np.meshgrid(self.Nc, self.Nr, indexing='xy')


        Nout = self.FTpad_outsize 
        Nin  = img_size 
        self.ratio_Nout_Nin = Nout[0] * Nout[1] / (Nin[0] * Nin[1])

        center     = [ np.floor(Nin[0]/2), np.floor(Nin[1]/2)]
        centerout  = [ np.floor(self.FTpad_outsize[0]/2), np.floor(self.FTpad_outsize[1]/2)]
        cenout_cen = [centerout[0] - center[0], centerout[1] - center[1]]

        idxs_1 = np.arange( max( cenout_cen[0],0), min( cenout_cen[0]+Nin[0],Nout[0]) )
        idxs_2 = np.arange( max( cenout_cen[1],0), min( cenout_cen[1]+Nin[1],Nout[1]) )
        idxs_3 = np.arange( max(-cenout_cen[0],0), min(-cenout_cen[0]+Nout[0],Nin[0]) )
        idxs_4 = np.arange( max(-cenout_cen[1],0), min(-cenout_cen[1]+Nout[1],Nin[1]) )
        
        self.xvi, self.yvi = np.meshgrid(idxs_1, idxs_2, indexing='ij')
        self.xvi, self.yvi = self.xvi.astype(np.int64), self.yvi.astype(np.int64)
        self.xvo, self.yvo = np.meshgrid(idxs_3, idxs_4, indexing='ij')
        self.xvo, self.yvo = self.xvo.astype(np.int64), self.yvo.astype(np.int64)

        self.row_shift = 0
        self.col_shift = 0


    def get_shifts(self):
        '''Get the last calculated shifts.\n
        ----------------------------------------------------
            `Returns:`
                img_shifts: [row_shift (float), col_shift (float)].\n   
        '''
        return [self.row_shift, self.col_shift]
    

    def FTpad(self, imFT):
        """ Pads or crops the Fourier transform to the desired ouput size. 
            Taking care that the zero frequency is put in the correct place for the output for subsequent FT or IFT. 
            Can be used for Fourier transform based interpolation, i.e. dirichlet kernel interpolation."""
        imFT = np.fft.fftshift(imFT)
        imFTout = np.zeros( (self.FTpad_outsize[0], self.FTpad_outsize[1]), dtype=imFT.dtype)
        imFTout[ self.xvi, self.yvi] = imFT[ self.xvo, self.yvo]
        imFTout = np.fft.ifftshift(imFTout) * self.ratio_Nout_Nin

        return imFTout

        
    def dftups(self, inp, roff, coff):
        """ Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just a small region.\n 
        roff, coff = Row and column offsets, allow to shift the output array to a region of interest on the DFT (default = 0)"""
        kernc=np.exp( self.const_nc * ( self.const_range_nc - coff  ) )
        kernr=np.exp( self.const_nr * np.expand_dims( self.const_range_nr - roff, axis=1) )

        return np.matmul(np.matmul(kernr,inp),kernc)
        

    def img_shift(self, image, shift=[0,0]):
        '''Subpixel-XY image shift.\n
        ----------------------------------------------------
            `Parameters:`
                image: (2D array) Image to be shifted.\n
                shift: [Row_shift (float), Col_shift (float)] Row and column subpixel shift.\n

            `Returns:`
                img_shifted: (2D array) Shifted image.\n   
        '''
        row_shift, col_shift = shift
        buf2ft = np.fft.fftn(image)
        img_shifted = np.abs( np.fft.ifftn( 
            buf2ft * np.exp( self.cx_2pi * (-row_shift*self.Nrf/self.nr-col_shift*self.Ncf/self.nc) ) 
                            ))
        return img_shifted


    def __call__(self, fixed, moving):
        '''Perform the DFT image registration.\n
        ----------------------------------------------------
            `Parameters:`
                fixed: (2D array) Reference image.\n
                moving: (2D array) Image to be registered.\n

            `Returns:`
                img_shifts: [row_shift (float), col_shift (float)].\n 
                registered: (2D array) Registered image.\n
        '''

        buf1ft = np.fft.fftn(fixed)
        buf2ft = np.fft.fftn(moving)

        buf_conj = buf1ft * np.conj(buf2ft)

        nr, nc = buf1ft.shape
        usfac = self.usfac
        # Single pixel registration
        if self.usfac == 1:
            CCabs = np.abs( np.fft.ifftn( buf_conj ) ) 
            max_idx = np.argmax(CCabs)
            row_shift = int(max_idx / nc)
            col_shift = max_idx % nc
            CCmax = CCabs[row_shift,col_shift] * self.total_num_pixels
            row_shift = self.Nr[row_shift]
            col_shift = self.Nc[col_shift]

        # Start with usfac == 2    
        elif self.usfac > 1:
            CC = np.fft.ifftn( self.FTpad( buf_conj ) )
            CCabs = np.abs(CC)
            shift = np.where( CCabs == np.max(CCabs) )
            row_shift = shift[0][0]
            col_shift = shift[1][0]
            CCmax = CC[row_shift, col_shift] * self.total_num_pixels
            row_shift = self.Nr2[row_shift]/2
            col_shift = self.Nc2[col_shift]/2
            
            # If upsampling > 2, then refine estimate with matrix multiply DFT
            if self.usfac > 2:
                row_shift = np.round(row_shift*usfac)/usfac
                col_shift = np.round(col_shift*usfac)/usfac
                CC = np.conj( self.dftups( buf2ft*np.conj(buf1ft),
                                                    self.dftshift-row_shift*self.usfac, 
                                                    self.dftshift-col_shift*self.usfac ) )
                # Locate maximum and map back to original pixel grid
                CCabs = np.abs(CC)
                shift = np.where( CCabs == np.max(CCabs) )
                rloc = shift[0][0]
                cloc = shift[1][0]
                CCmax = CC[rloc,cloc]
                rloc = rloc - self.dftshift
                cloc = cloc - self.dftshift
                row_shift = row_shift + rloc/usfac
                col_shift = col_shift + cloc/usfac

        self.row_shift = row_shift 
        self.col_shift = col_shift 

        # Compute registered version of the moving image
        diffphase = np.angle(CCmax)
        Greg = buf2ft*np.exp( self.cx_2pi * (-row_shift*self.Nrf/self.nr-col_shift*self.Ncf/self.nc) )
        Greg = Greg*np.exp(1j*diffphase)

        # Return the image shifts and the registered image
        return [row_shift,col_shift], np.abs( np.fft.ifftn(Greg) )
    



# PyTorch-based DFT-based subpixel image registration class
class Torch_DFT_Registrator():
    '''
    PyTorch-based subpixel shift estimation and Image registration.
    '''

    def __init__(self, img_size, usfac=20, device="cpu"):
        '''DFT-based subpixel shift estimation and Image registration.\n
        ----------------------------------------------------
            `Parameters`:\n
                img_size: [Rows, Cols] Number of rows and columns of the image.\n
                usfac: Upsampling factor. The image shifts and registeration \
                       will be computed to 1/usfac fraction of a pixel.\n
                device: The device to run computations on ("cpu" or "cuda").
        '''
        nr, nc = img_size
        self.usfac = usfac
        self.device = device
        self.pi = torch.pi
        imag_part = torch.tensor(2 * self.pi).to(self.device)
        real_part = torch.zeros_like(imag_part)
        self.cx_2pi = torch.complex(real_part, imag_part).to(torch.complex64)
        
        self.nr = torch.tensor(nr, device=self.device)
        self.nc = torch.tensor(nc, device=self.device)
        self.FTpad_outsize = torch.tensor([2 * nr, 2 * nc], device=self.device)
        
        self.total_num_pixels = self.nr * self.nc
        self.dftshift = torch.fix(torch.ceil(torch.tensor(self.usfac * 1.5)) / 2).to(self.device)

        # Constant parameters for the function "dftups"
        self.nor = torch.ceil(torch.tensor(usfac * 1.5, device=self.device))
        self.noc = torch.ceil(torch.tensor(usfac * 1.5, device=self.device))
        
        # Ensure the arange result is on the device before operations
        self.const_nc = ( -self.cx_2pi / (self.nc * self.usfac) ) * \
                        ( torch.fft.ifftshift(torch.arange(self.nc, device=self.device) - torch.floor(self.nc / 2)).unsqueeze(1) )
        self.const_nr = ( -self.cx_2pi / (self.nr * self.usfac) ) * \
                        ( torch.fft.ifftshift(torch.arange(self.nr, device=self.device)) - torch.floor(self.nr / 2) )

        self.const_range_nc = torch.arange(self.noc.item(), device=self.device)
        self.const_range_nr = torch.arange(self.nor.item(), device=self.device)
        
        self.Nr = torch.fft.ifftshift(torch.arange(-nr // 2, (nr + 1) // 2, dtype=torch.int, device=self.device))
        self.Nc = torch.fft.ifftshift(torch.arange(-nc // 2, (nc + 1) // 2, dtype=torch.int, device=self.device))
        self.Nr2 = torch.fft.ifftshift(torch.arange(-nr, nr, dtype=torch.int, device=self.device))
        self.Nc2 = torch.fft.ifftshift(torch.arange(-nc, nc, dtype=torch.int, device=self.device))
        [self.Ncf, self.Nrf] = torch.meshgrid(self.Nc, self.Nr, indexing='xy')

        Nout = self.FTpad_outsize
        Nin = torch.tensor(img_size, device=self.device)
        self.ratio_Nout_Nin = Nout.prod() / Nin.prod()

        center = [torch.floor(Nin[0] / 2), torch.floor(Nin[1] / 2)]
        centerout = [torch.floor(self.FTpad_outsize[0] / 2), torch.floor(self.FTpad_outsize[1] / 2)]
        cenout_cen = [centerout[0] - center[0], centerout[1] - center[1]]

        idxs_1 = torch.arange(max(cenout_cen[0], 0), min(cenout_cen[0] + Nin[0], Nout[0]), device=self.device)
        idxs_2 = torch.arange(max(cenout_cen[1], 0), min(cenout_cen[1] + Nin[1], Nout[1]), device=self.device)
        idxs_3 = torch.arange(max(-cenout_cen[0], 0), min(-cenout_cen[0] + Nout[0], Nin[0]), device=self.device)
        idxs_4 = torch.arange(max(-cenout_cen[1], 0), min(-cenout_cen[1] + Nout[1], Nin[1]), device=self.device)
        
        self.xvi, self.yvi = torch.meshgrid(idxs_1, idxs_2, indexing='ij')
        self.xvi, self.yvi = self.xvi.long(), self.yvi.long()
        self.xvo, self.yvo = torch.meshgrid(idxs_3, idxs_4, indexing='ij')
        self.xvo, self.yvo = self.xvo.long(), self.yvo.long()

        self.row_shift = 0.0
        self.col_shift = 0.0


    def get_shifts(self):
        '''Get the last calculated shifts.\n
        ----------------------------------------------------
            `Returns:`
                img_shifts: [row_shift (float), col_shift (float)].\n   
        '''
        return [self.row_shift, self.col_shift]
    

    def FTpad(self, imFT):
        """Pads or crops the Fourier transform to the desired output size."""
        imFT = torch.fft.fftshift(imFT, dim=(-2, -1))
        imFTout = torch.zeros(
            (self.FTpad_outsize[0], self.FTpad_outsize[1]),
            dtype=imFT.dtype,
            device=self.device,
        )
        imFTout[self.xvi, self.yvi] = imFT[self.xvo, self.yvo]
        imFTout = torch.fft.ifftshift(imFTout, dim=(-2, -1)) * self.ratio_Nout_Nin
        return imFTout


    def dftups(self, inp, roff, coff):
        """ Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just a small region.\n 
        roff, coff = Row and column offsets, allow to shift the output array to a region of interest on the DFT (default = 0)"""

        kernc = torch.exp( self.const_nc * ( self.const_range_nc - coff  ).unsqueeze(0) )
        kernr = torch.exp( self.const_nr.unsqueeze(0) * ( self.const_range_nr - roff ).unsqueeze(1) )

        return torch.matmul(torch.matmul(kernr, inp), kernc)


    def img_shift(self, image, shift=[0.0, 0.0]):
        '''Subpixel-XY image shift.\n
        ----------------------------------------------------
            `Parameters:`
                image: (2D array) Image to be shifted.\n
                shift: [Row_shift (float), Col_shift (float)] Row and column subpixel shift.\n

            `Returns:`
                img_shifted: (2D array) Shifted image.\n   
        '''
        with torch.no_grad():
            row_shift, col_shift = shift
            # Convert NumPy input image to PyTorch Tensor
            image_t = torch.tensor(image, dtype=torch.float32, device=self.device)
            buf2ft = torch.fft.fftn(image_t)
            # Corrected phase ramp for a positive shift
            cx_neg_2pi = (
                torch.tensor(-1j * 2 * torch.pi, dtype=torch.complex64)
                .to(self.device)
            )
            
            phase_ramp_arg = (
                cx_neg_2pi 
                * (
                    (self.Nrf.float() * row_shift / self.nr)
                    + (self.Ncf.float() * col_shift / self.nc)
                )
            )
            
            shifted_ft = buf2ft * torch.exp(phase_ramp_arg)
            img_shifted_t = torch.abs(torch.fft.ifftn(shifted_ft))
            
            return img_shifted_t.cpu().numpy()

        
    def __call__(self, fixed, moving):
        '''Perform the DFT image registration.
            Parameters:
                fixed: (2D array) Reference image.
                moving: (2D array) Image to be registered.
            Returns:
                img_shifts: [row_shift (float), col_shift (float)].
                registered: (2D array) Registered image.
        '''
        with torch.no_grad():
            # Convert NumPy inputs to PyTorch tensors
            fixed_t = torch.tensor(fixed, dtype=torch.float32, device=self.device)
            moving_t = torch.tensor(moving, dtype=torch.float32, device=self.device)
            
            buf1ft = torch.fft.fftn(fixed_t)
            buf2ft = torch.fft.fftn(moving_t)

            buf_conj = buf1ft * torch.conj(buf2ft)
            
            # Single pixel registration ---
            if self.usfac == 1:
                CCabs = torch.abs(torch.fft.ifftn(buf_conj))
                max_idx = torch.argmax(CCabs)
                
                # Use standard tensor indexing for 2D array
                row_idx = max_idx // self.nc.long()
                col_idx = max_idx % self.nc.long()
                
                row_shift = self.Nr[row_idx]
                col_shift = self.Nc[col_idx]
                CCmax = CCabs.flatten()[max_idx] * self.total_num_pixels
            
            #  Subpixel registration 
            else:
                CC = torch.fft.ifftn(self.FTpad(buf_conj))
                CCabs = torch.abs(CC)
                shift = torch.where(CCabs == torch.max(CCabs))
                
                row_shift = self.Nr2[shift[0][0]] / 2
                col_shift = self.Nc2[shift[1][0]] / 2
                CCmax = CC[shift[0][0], shift[1][0]] * self.total_num_pixels
                
                # Refine estimate if upsampling > 2
                if self.usfac > 2:
                    row_shift_round = torch.round(row_shift * self.usfac) / self.usfac
                    col_shift_round = torch.round(col_shift * self.usfac) / self.usfac
                    
                    # Compute DFT over small region for refinement
                    CC = torch.conj(self.dftups(
                        buf2ft * torch.conj(buf1ft),
                        self.dftshift - row_shift_round * self.usfac, 
                        self.dftshift - col_shift_round * self.usfac
                    ))
                    
                    CCabs = torch.abs(CC)
                    shift = torch.where(CCabs == torch.max(CCabs))
                    
                    # Locate maximum and map back to original pixel grid
                    rloc = shift[0][0] - self.dftshift
                    cloc = shift[1][0] - self.dftshift
                    CCmax = CC[shift[0][0], shift[1][0]]
                    
                    row_shift = row_shift_round + rloc / self.usfac
                    col_shift = col_shift_round + cloc / self.usfac

            # Store shifts and perform final registration
            self.row_shift = row_shift.item()
            self.col_shift = col_shift.item()

            diffphase = torch.angle(CCmax)
            
            # Final registration exponential term
            Greg = buf2ft * torch.exp(self.cx_2pi * (-row_shift * self.Nrf.float() / self.nr - col_shift * self.Ncf.float() / self.nc))
            Greg = Greg * torch.exp(1j * diffphase)

            # Return shifts and registered image as NumPy arrays
            return [self.row_shift, self.col_shift], torch.abs(torch.fft.ifftn(Greg)).cpu().numpy()




