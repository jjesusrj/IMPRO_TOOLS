from tkinter import Image
from turtle import width
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
import cv2
from typing import Union, Tuple


class PatternMaker:
    sharp_kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    
    def __init__(self):
        pass  # No initialization needed for now


    @staticmethod
    def create_checkerboard(size:tuple=(800, 800), block_size:int=50) -> np.ndarray:
        """
        Creates a black and white checkerboard image.

        Args:
            size (tuple): Size of the image in pixels (height, width).
            block_size (int): Size of each square block.

        Returns:
            image (np.ndarray): Checkerboard pattern image in BGR format.
        """
        img = np.zeros(size, dtype=np.uint8)
        for y in range(0, size[0], block_size):
            for x in range(0, size[1], block_size):
                if (x // block_size) % 2 == (y // block_size) % 2:
                    img[y:y+block_size, x:x+block_size] = 255

        # Convert to BGR for compatibility with color drawing
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    


    @staticmethod
    def make_turing(image: Image.Image, rep: int = 20, radius: int = 5, 
                    sharpen_percent: int = 300, as_rgb: bool = False
                    ) -> Union[Image.Image, np.ndarray]:
        """
        Applies Turing pattern effect to an existing image.
        Args:
            image (PIL.Image): The input image.
            rep (int): Number of times to apply the blur and sharpen filters.
            radius (int): Radius for the blur and sharpen filters.
            sharpen_percent (int): Percent for the sharpen filter. """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        for _ in range(rep):
            img = img.filter(ImageFilter.BoxBlur(radius=radius))
            img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen_percent, threshold=0))
        if as_rgb:
            img = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return img
    


    @classmethod
    def make_turing_cv2(cls, image: Union[Image.Image, np.ndarray], rep: int = 20, radius: int = 5, 
                        sharpen_percent: int = 300, as_rgb: bool = False, as_PIL: bool = True
                        ) -> Union[Image.Image, np.ndarray]:
        """
        Implementation using OpenCV that is almost equivalent to the original Pillow code.
        """
        
        # Setup and Type Conversion
        if isinstance(image, Image.Image):
            # Convert to grayscale float32
            img_np = np.array(image.convert('L'), dtype=np.float32)
        elif isinstance(image, np.ndarray):
            img_np = image 
            if img_np.ndim == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            img_np = img_np.astype(np.float32) 
        else:
            raise TypeError("Input 'image' must be a PIL.Image or a numpy.ndarray.")

        # Kernel and Sigma Calculation
        # Box Blur Kernel Size (Using INT(radius) for sharpness)
        r_int = max(1, int(radius)) 
        box_ksize_int = 2 * r_int + 1
        
        # Gaussian Sigma Calculation (Matches Pillow's UnsharpMask internal sigma)
        gaussian_sigma = 0.4 * float(radius) + 0.6
        
        # Sharpening strength (alpha = percent / 100)
        alpha = sharpen_percent / 100.0

        ## Blur and Sharpen
        for _ in range(rep):
            # Box Blur 
            blurred_base = cv2.boxFilter(
                img_np, 
                ddepth=-1, 
                ksize=(box_ksize_int, box_ksize_int), 
                normalize=True
            )
            # Create the mask (The Gaussian blur of the blurred image)
            gaussian_blurred = cv2.GaussianBlur(blurred_base, ksize=(0, 0), sigmaX=gaussian_sigma)
            # Calculate the Mask (Detail = Base - Gaussian Blurred)
            mask = blurred_base - gaussian_blurred
            # Apply Sharpening: Base + alpha * Mask
            img_np = np.clip(blurred_base + alpha * mask, 0, 255)

        output_array = img_np.astype(np.uint8)
        if as_rgb:
            output_array = cv2.cvtColor(output_array, cv2.COLOR_GRAY2RGB)
        if as_PIL:
            return Image.fromarray(output_array)
        else:
            return output_array



    @classmethod
    def create_turing_pattern(cls, size: Tuple[int, int], rep: int = 20, 
                              radius: int = 5, sharpen_percent: int = 300) -> Union[Image.Image, np.ndarray]:
        """
        Creates a random Turing pattern image.
        Args:
            size (tuple): Size of the image in pixels (height, width).
            rep (int): Number of times to apply the blur and sharpen filters.
            radius (int): Radius for the blur and sharpen filters.
            sharpen_percent (int): Percent for the sharpen filter. """
        img = (np.random.random(size)*255).astype(np.uint8)
        return cls.make_turing(img, rep, radius, sharpen_percent)
    


    @classmethod
    def create_turing_image(cls,image: np.ndarray, rep: int = 20, radius: int = 1, sharpen_percent: int = 200,
                            canny_th: Tuple[int, int] = (100, 200), as_rgb: bool = False, use_cv2: bool = False,
                             as_PIL: bool = False) -> Union[Image.Image, np.ndarray]:
        """
        Creates a Turing pattern image from the edges of an input image.

        Args:
            image (np.ndarray): Input image (grayscale or BGR color).
            rep (int): Number of times to apply blur and sharpen filters.
            radius (int): Radius for blur and sharpen filters.
            sharpen_percent (int): Percent for the sharpen filter.
            canny_th (tuple[int, int]): Thresholds for Canny edge detection.
            as_rgb (bool): Convert output to RGB if True.
            use_cv2 (bool): Use OpenCV implementation of Turing effect if True.
            as_PIL (bool): If True and use_cv2=True, return a PIL.Image; otherwise, return np.ndarray.

        Returns:
            PIL.Image.Image or np.ndarray: Turing pattern image.
        """
        if len(image.shape)==3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply Canny edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        blurred = gray
        edges = 255 - cv2.Canny(blurred, canny_th[0], canny_th[1])
        if use_cv2:
            return cls.make_turing_cv2(edges, rep, radius, sharpen_percent, as_rgb, as_PIL)
        else:
            return cls.make_turing(edges, rep, radius, sharpen_percent, as_rgb)
    


    @staticmethod
    def draw_circle(image: Image.Image, x_coord: int, y_coord: int, fill: str = 'black',
                    outline: str = 'white',circle_diameter: int = 10,line_width: int = 1) -> None:
        """
        Draws a circle on the given image.

        Args:
            image (PIL.Image.Image): The image to draw on.
            x_coord (int): The x-coordinate of the circle's center.
            y_coord (int): The y-coordinate of the circle's center.
            fill (str): The fill color of the circle.
            outline (str): The outline color of the circle.
            circle_diameter (int): The diameter of the circle.
            line_width (int): The width of the outline.
        """
        draw = ImageDraw.Draw(image)
        circle_radius = circle_diameter/2
        x1, x2 = x_coord-circle_radius, x_coord+circle_radius
        y1, y2 = y_coord-circle_radius, y_coord+circle_radius
        draw.ellipse((x1-line_width, y1-line_width, x2+line_width, y2+line_width), fill=outline)
        draw.ellipse((x1, y1, x2, y2), fill=fill)



    @staticmethod
    def draw_rect(image: Image.Image, rect: Tuple[int, int, int, int],fill: str = 'black',
                  outline: str = 'white',line_width: int = 1) -> None:
        """
        Draws a rectangle on the given image.

        Args:
            image (PIL.Image.Image): The image to draw on.
            rect (tuple[int, int, int, int]): The rectangle defined as (left, top, right, bottom).
            fill (str): The fill color of the rectangle.
            outline (str): The outline color of the rectangle.
            line_width (int): The width of the outline.
        """
        draw = ImageDraw.Draw(image)
        draw.rectangle(rect, fill=fill, outline=outline, width=line_width)



    @staticmethod
    def draw_triangle(image: Image.Image,rect: Tuple[int, int, int, int], fill: str = 'black',
                      outline: str = 'white',line_width: int = 1) -> None:
        """
        Draws a triangle inside the given bounding rectangle on the image.

        Args:
            image (PIL.Image.Image): The image to draw on.
            rect (tuple[int, int, int, int]): The bounding rectangle defined as (left, top, right, bottom).
            fill (str): The fill color of the triangle.
            outline (str): The outline color of the triangle.
            line_width (int): The width of the outline.

        Returns:
            None
        """
        draw = ImageDraw.Draw(image)
        points = [  (rect[0],rect[3]),
                    (rect[0],rect[1]),
                    (rect[2],rect[1])]
        draw.polygon(points, fill=fill, outline=outline, width=line_width)
