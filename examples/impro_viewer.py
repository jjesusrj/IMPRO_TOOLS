# Standard library imports
import os
import sys
import math
import time
import platform
# import warnings
import subprocess
from queue import Queue

# Third-party imports
import cv2
import numpy as np
from typing import Dict, Union, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Safe for headless environments
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pyqtgraph as pg
import qtawesome as qta

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QGraphicsView, QGraphicsScene,
    QMessageBox, QToolButton, QSizePolicy,
    QTableWidgetItem, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QPalette, QColor

# Add parent directory to the Python path
curr_dir = os.getcwd()
sys.path.append(curr_dir)
# Impoprt the image slicer for the ROI
from impro_tools.ImageSlicer import ImageSlicer
get_slice = ImageSlicer.slice_along_line    # Use a shorter name
# Import the contrast class for Michelson and MTF contrast
from impro_tools.Contrast import Contrast as ct

# Import the Turing filter
from impro_tools.PatternMaker import PatternMaker as pm

# Suppress Qt debug logs
os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.connect=false"

# Platform detection
OS_NAME = platform.system()
if OS_NAME == "Linux":
    print("Running on Linux")
elif OS_NAME == "Darwin":
    print("Running on macOS")
elif OS_NAME == "Windows":
    print("Running on Windows")
    raise NotImplementedError(f"Unsupported OS: {OS_NAME}")

# Constants
SAT_THRESHOLD   = 252        # Oversaturation threshold (99% of 8-bit DR)
PIX_SIZE        = 0.285      # Pixel size for Standard cameras
ZERO_DIV_CONST  = 1e-13      # Small constant to avoid division by zero


# Get the OS name
OS_NAME = platform.system()










def list_cameras() -> Dict[str, Union[int, str]]:
    """
    List available camera devices on the system.

    On **Linux**: uses `v4l2-ctl --list-devices` to detect `/dev/video*` devices.

    On **macOS**: tries opening camera indices with OpenCV to see which ones are available.

    Returns
    -------
    Dict[str, Union[int, str]]
        A dictionary mapping camera IDs (e.g. "cam0", "cam1", ...) to either:
        - (str) device path (str) on Linux, e.g. ["/dev/video0", "/dev/video1", ...]
        - (int) index for OpenCV capture on macOS [0, 1, ...]
    Raises
    ------
    NotImplementedError
        If the OS is not supported.
    """
    serials: Dict[str, Union[int, str]] = {}
    os_name = platform.system()

    # ---------- Linux ----------
    if os_name == "Linux":
        # Run v4l2-ctl to list devices. The output contains device names and their /dev/video paths.
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
        )
        output = result.stdout.replace('\t', '').splitlines()
        cam_idx = 0
        print("Looking for cameras (Linux):")
        for line in output:
            line = line.strip()
            if line.startswith("/dev/video"):
                device = line
                cam_id = f"cam{cam_idx}"
                serials[cam_id] = device
                print(f" - Found camera {cam_id}: {device}")
                cam_idx += 1
        return serials

    # ---------- macOS (Darwin) ----------
    elif os_name == "Darwin":
        max_tested = 10
        print("Looking for cameras (macOS):")
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                cam_id = f"cam{i}"
                serials[cam_id] = i
                print(f" - Found camera {cam_id}: index {i}")
            else:
                cap.release()
                # If a camera index fails, assume further ones will too (optional logic)
                break
        return serials

    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")





# ====================================================================================
#                               Class PlotWidget
# ====================================================================================
class PlotWidget(QWidget):
    """
    A widget that displays two plots using pyqtgraph:
     1. A Modulation Transfer Function (MTF) vs. spatial frequency curve.
     2. An intensity profile (and fitted profile) of the Region of Interest (ROI).

    Allows updating of plotted data (frequency, MTF, profile) and resetting to default.
    """

    def __init__(self, parent=None):
        """
        Initialize the PlotWidget with its layout, plots, and reference lines.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)

        # First plot: MTF
        self.plot1 = self.graphics_layout.addPlot(title="ROI MTF")
        self.plot1.showGrid(x=True, y=True)
        self.plot1.setYRange(-0.1, 1.1)
        self.plot1.setLabel('bottom', "Spatial Frequency", units='Cycles/mm')
        self.plot1.setLabel('left', "Normalized MTF")
        self.plot1.addLegend(offset=(-10, 10), anchor=(1, 1))

        # Dummy initial data
        n_samples = 20
        y = np.zeros(n_samples + 1)
        x = np.arange(n_samples + 1)
        self.curve1 = self.plot1.plot(x, y, pen=pg.mkPen('y', width=2), name="MTF Curve")

        # Reference horizontal levels for MTF50 and MTF10
        self.mtf50_value = 0.0
        self.mtf10_value = 0.0
        self.mtf50_level = 0.5
        self.mtf10_level = 0.1

        # Horizontal threshold lines (fixed MTF values)
        self.hline_mtf50 = pg.InfiniteLine(
            pos=self.mtf50_level,
            angle=0,
            pen=pg.mkPen((255, 100, 100, 150), style=Qt.DashLine, width=1.5),
            label='MTF = 0.5',
            labelOpts={'position': 0.07, 'color': (255, 100, 100), 'fill': (20, 20, 20, 200)}
        )
        self.hline_mtf10 = pg.InfiniteLine(
            pos=self.mtf10_level,
            angle=0,
            pen=pg.mkPen((100, 255, 100, 150), style=Qt.DashLine, width=1.5),
            label='MTF = 0.1',
            labelOpts={'position': 0.07, 'color': (100, 255, 100), 'fill': (20, 20, 20, 200)}
        )

        # Vertical lines (for detected spatial frequencies MTF50 and MTF10)
        self.vline_mtf50 = pg.InfiniteLine(
            pos=self.mtf50_value,
            angle=90,
            pen=pg.mkPen((255, 100, 100, 200), style=Qt.DashLine, width=2),
            label='MTF50',
            labelOpts={'position': 0.9, 'color': (255, 100, 100), 'fill': (30, 30, 30, 200)}
        )
        self.vline_mtf10 = pg.InfiniteLine(
            pos=self.mtf10_value,
            angle=90,
            pen=pg.mkPen((100, 255, 100, 200), style=Qt.DashLine, width=2),
            label='MTF10',
            labelOpts={'position': 0.9, 'color': (100, 255, 100), 'fill': (30, 30, 30, 200)}
        )

        # Add all lines to the plot
        self.plot1.addItem(self.hline_mtf50)
        self.plot1.addItem(self.hline_mtf10)
        self.plot1.addItem(self.vline_mtf50)
        self.plot1.addItem(self.vline_mtf10)

        # ROI Profile Plot
        self.graphics_layout.nextRow()
        self.plot2 = self.graphics_layout.addPlot(title="ROI Profile")
        self.plot2.showGrid(x=True, y=True)
        self.plot2.setYRange(-0.1, 1.1)
        self.plot2.setLabel('bottom', "Pixel")
        self.plot2.setLabel('left', "Intensity")
        self.plot2.addLegend()
        self.curve2 = self.plot2.plot(x, y, pen=pg.mkPen('c', width=2), name="Profile")
        self.curve3 = self.plot2.plot(x, y, pen=pg.mkPen('m', width=2, style=Qt.DotLine), name="Fitted Profile")


    # Update function
    @Slot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float)
    def update_plot(self, f_pos: np.ndarray, mtf_norm: np.ndarray,
                    norm_profile: np.ndarray, y_data_fitted: np.ndarray,
                    mtf50: float, mtf10: float):
        """
        Update the data on the plots.

        Parameters
        ----------
        f_pos : np.ndarray
            Spatial frequency axis for MTF (e.g. cycles/mm).
        mtf_norm : np.ndarray
            Normalized MTF values corresponding to f_pos.
        norm_profile : np.ndarray
            The normalized ROI intensity profile.
        y_data_fitted : np.ndarray
            The fitted profile (e.g. from a tanh or other model).
        mtf50 : float
            The spatial frequency where MTF falls to 0.5.
        mtf10 : float
            The spatial frequency where MTF falls to 0.1.
        """
        # Update MTF curve
        if f_pos is not None:
            self.curve1.setData(f_pos, mtf_norm)
        self.curve2.setData(np.arange(len(norm_profile)), norm_profile)
        self.curve3.setData(np.arange(len(y_data_fitted)), y_data_fitted)

        # Update vertical lines (MTF50/MTF10 frequency positions)
        self.vline_mtf50.setValue(mtf50)
        self.vline_mtf10.setValue(mtf10)

        # (Horizontal lines remain fixed at y=0.5, 0.1)
        self.mtf50_value = mtf50
        self.mtf10_value = mtf10


    def reset_plot(self):
        """
        Reset all plots and lines to their initial (zeroed) state.
        """
        n_samples = 20
        y = np.zeros(n_samples + 1)
        x = np.arange(n_samples + 1)
        # Reset MTF curve
        self.curve1.setData(x, y)
        self.curve2.setData(x,y)
        self.curve3.setData(x,y)

        # Update vertical lines (MTF50/MTF10 frequency positions)
        self.vline_mtf50.setValue(0)
        self.vline_mtf10.setValue(0)

        # Set the vertical lines to 0. the Horizontal lines remain fixed at y=0.5, 0.1
        self.mtf50_value = 0
        self.mtf10_value = 0






# ====================================================================================
#                                    Class Camera
# ====================================================================================
class Camera:
    """
    A simple camera wrapper around OpenCV's VideoCapture for cross‑platform use.
    """

    def __init__(
        self,
        serial: str = "0000000000000000",
        device: Union[str, int] = "/dev/video0",
        exposure: float = 10000.0,
        img_format: str = "UYVY",
        width: int = 1344,
        height: int = 1344
    ) -> None:
        """
        Initialize the Camera object.

        Parameters
        ----------
        serial : str
            The serial number of the camera.
        device : Union[str, int]
            Device path or index for OpenCV VideoCapture.
        exposure : float
            Initial exposure value.
        img_format : str
            FourCC image format.
        width : int
            Desired capture width.
        height : int
            Desired capture height.
        """
        self.serial = serial
        self.device = device
        self.width = width
        self.height = height
        self.img_format = img_format
        self.exposure = exposure

        self.cap: Optional[cv2.VideoCapture] = None
        self.connected: bool = False

    def connect(self) -> bool:
        """
        Open the camera and configure capture settings.

        Returns
        -------
        bool
            True if the camera was successfully opened, False otherwise.
        """
        system_name = platform.system()
        print(f"- Opening device: {self.device} | Serial: {self.serial} on {system_name}")

        # Choose API/backend based on OS
        if system_name == "Linux":
            # Use V4L2 backend on Linux
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            # Set manual exposure mode (V4L2)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
            # Set fourcc / pixel format
            if self.img_format == "UYVY":
                fourcc = cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y')
            elif self.img_format == "BG12":
                fourcc = cv2.VideoWriter_fourcc('B', 'G', '1', '2')
            else:
                fourcc = 0
            if fourcc != 0:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # Read back what format was actually set
            actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
            if codec != self.img_format:
                print(f"   WARNING: unable to set image format to {self.img_format}. Using [{codec}].")

        elif system_name == "Darwin":
            # Use AVFoundation on macOS
            cap = cv2.VideoCapture(self.device, cv2.CAP_AVFOUNDATION)
            # Try to disable auto-exposure
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        else:
            raise NotImplementedError(f"Unsupported OS: {system_name}")

        self.cap = cap

        if not cap.isOpened():
            print(f"Failed to open camera device: {self.device}")
            return False

        # Set frame size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Optionally, set exposure after size
        # cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

        self.connected = True
        return True

    def get_exposure(self) -> float:
        """
        Get the current exposure setting from the camera.

        Returns
        -------
        Optional[float]
            The exposure value, or None if camera is not connected.
        """
        if self.cap is None:
            print("Warning: the exposure time could not be read.")
            return 0.0
        return self.cap.get(cv2.CAP_PROP_EXPOSURE)

    def set_exposure(self, new_exposure: float) -> bool:
        """
        Set a new exposure value for the camera.

        Parameters
        ----------
        new_exposure : float
            The desired exposure.

        Returns
        -------
        bool
            True if setting succeeded, False otherwise.
        """
        if self.cap is None:
            return False
        return self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)

    def disconnect(self) -> None:
        """
        Release the camera resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.connected = False





# ====================================================================================
#                                 Class CameraThread
# ====================================================================================
class CameraThread(QThread):
    """
    A QThread subclass that continuously reads frames from a Camera,
    measures the frame rate (FPS), and emits the current FPS via a signal.
    Frames optionally get pushed into a queue.
    """

    frame_rate = Signal(str)

    def __init__(self, camera: "Camera", frame_queue: Queue) -> None:
        """
        Initialize the CameraThread.

        Parameters
        ----------
        camera : (Camera) The Camera object from which to grab frames.
        frame_queue : (Queue) A queue to put raw frames into, by default None.
        """
        super().__init__()
        self.camera = camera
        self.frame_queue = frame_queue
        self.running = True
        self.streaming = False
        # FPS measurement
        self._last_time = time.time()
        self._frame_count = 0
        self.fps = 0.0


    def run(self) -> None:
        """
        The thread's main loop: if streaming is enabled, capture frames
        from the camera, push to queue, and emit frame rate every second.
        """
        print(f"Camera {self.camera.serial} streaming thread started.")
        self._last_time = time.time()

        while self.running:
            if self.streaming and self.camera.cap is not None:
                ret, frame = self.camera.cap.read()
                if not ret:
                    print("CameraThread: Failed to read frame.")
                    continue

                if frame.dtype == np.uint16:
                    frame= (frame / 256).astype(np.uint8) 
                
                # Update FPS counter
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_time
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    fps_text = f"FPS: {int(np.round(self.fps)):3}"
                    self.frame_rate.emit(fps_text)
                    self._frame_count = 0
                    self._last_time = now

                # Push frame to queue if provided
                if self.frame_queue is not None and not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                # Yield control to avoid max CPU usage
                QThread.msleep(1) # could be used if desired

        print(f"Camera {self.camera.serial} streaming thread stopped.")


    def set_camera(self, camera: "Camera") -> None:
        """
        Change the Camera object this thread is using.

        Parameters
        ----------
        camera : Camera
        """
        self.camera = camera


    def start_streaming(self) -> None:
        """
        Enable the streaming of frames in the thread.
        """
        self.streaming = True


    def stop_streaming(self) -> None:
        """
        Disable the streaming of frames in the thread.
        """
        self.streaming = False


    def set_exposure(self, new_exposure: float) -> None:
        """
        Change the camera's exposure via this thread.

        Parameters
        ----------
        new_exposure : float
        """
        if self.camera.cap is not None:
            successful = self.camera.set_exposure(new_exposure)
            if not successful:
                print(f"CameraThread: Failed to set exposure to {new_exposure}")


    def stop(self) -> None:
        """
        Stop the thread gracefully. After calling, wait() on the thread
        in the caller to ensure it finishes.
        """
        self.running = False
        self.wait()








# ====================================================================================
#                                Class ProcessingThread
# ====================================================================================
class ProcessingThread(QThread):
    """
    Thread used for processing incoming frames from a queue. The images are processed 
    differetn operations such as oversaturation, drawing grid/ROI overlays, calculating 
    contrast/MTF, and emitting signals with processed data.

    Parameters
    ----------
    frame_queue : (Queue) the queue containing incoming image frames to be processed.
    """

    img_processed   = Signal(np.ndarray)
    show_plot       = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float)
    show_mtf        = Signal(np.ndarray)
    show_mcontrast  = Signal(float)
    show_saturation = Signal(float)
    show_angle      = Signal(float, float)

    def __init__(self, frame_queue:Queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.current_frame = None   # The current raw frame
        self.proc_frame = None      # The current processed frame
        self.running = True

        # Flags to control what process to excecute
        self.update = False
        self.do_mtf = False
        self.do_mcontrast = False
        self.do_turing = False
        self.show_grid = False
        
        # ROI and grid settings
        self.p1, self.p2  = (0, 0), (35, 35)    # ROI points
        self.roi_alpha = 0.5      # Rectangle ROI transparency
        self.grid_alpha = 0.1     # Grid transparency
        self.grid_spacing = 50    # Grid spacing
        self.grid = None

    def set_roi_points(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """
        Set ROI endpoints used for slicing and contrast/MTF calculations.

        Parameters
        ----------
        p1 : tuple(int, int)
            Starting point of the ROI.
        p2 : tuple(int, int)
            Ending point of the ROI.
        """
        self.p1, self.p2 = p1, p2
        self.get_angle()

    def run(self):
        """
        Main thread loop:
        - Pulls frames from queue.
        - Computes oversaturation mask.
        - Draws grid and ROI overlays.
        - Computes Michelson contrast or MTF if enabled.
        - Emits processed frames and results.
        """
        while self.running:
            if self.frame_queue is not None and not self.frame_queue.empty():
                self.current_frame = self.frame_queue.get()
                self.update = True

            if self.update and self.current_frame is not None:
                self.proc_frame = self.current_frame.copy()

                if self.do_turing:
                    # cv2.imwrite("example_img.png", self.current_frame)
                    self.proc_frame = pm.create_turing_image(self.proc_frame, rep=20, radius=2, 
                                                             sharpen_percent=300, canny_th=(100,100), 
                                                             as_rgb=True, use_cv2=True, as_PIL=False)
                    # Emit processed frame
                    if self.proc_frame is not None:
                        self.img_processed.emit(self.proc_frame)
                    self.update = False
                    continue

                # Calculate oversaturation of the image
                saturation_mask = np.any(self.current_frame >= SAT_THRESHOLD, axis=2)
                saturation = np.sum(saturation_mask) / saturation_mask.size

                if saturation >= 0.001:
                    self.proc_frame[saturation_mask] = [0, 0, 255]
                    self.show_saturation.emit(saturation)
                else:
                    self.show_saturation.emit(0)

                if self.show_grid or self.do_mcontrast or self.do_mtf:
                    if self.show_grid:
                        if self.grid is None:
                            self.grid = np.zeros_like(self.proc_frame)
                            for x in range(0, 1344, self.grid_spacing):
                                cv2.line(self.grid, (x, 0), (x, 1344), (255, 255, 255), 2)
                                cv2.line(self.grid, (0, x), (1344, x), (255, 255, 255), 2)
                        cv2.addWeighted(self.grid, self.grid_alpha, self.proc_frame, 
                                        1 - self.grid_alpha, 0, self.proc_frame)

                    # ROI extraction
                    roi, roi_corners = get_slice(
                        cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY),
                        self.p1, self.p2, 20
                    )
                    roi_corners = np.array(roi_corners, dtype=np.int32)

                    # Draw overlays
                    overlay = self.proc_frame.copy()
                    cv2.polylines(overlay, [roi_corners], True, (0, 0, 255), 1)

                    if roi_corners is not None and len(roi_corners) == 4:
                        cv2.line(overlay,
                                 (roi_corners[1] + roi_corners[2]) // 2,
                                 (roi_corners[3] + roi_corners[0]) // 2,
                                 (255, 255, 0), 1)

                    cv2.line(overlay, self.p1, self.p2, (0, 255, 0), 1)
                    cv2.circle(overlay, self.p1, 3, (255, 255, 0), -1)
                    cv2.circle(overlay, self.p2, 3, (255, 0, 0), -1)

                    cv2.addWeighted(overlay, self.roi_alpha,
                                    self.proc_frame, 1 - self.roi_alpha,
                                    0, self.proc_frame)

                    # Get profile
                    if roi is None:
                        profile = np.zeros(10)
                    else:
                        profile = np.mean(roi, axis=0)

                    norm_profile = profile / (profile.max() + ZERO_DIV_CONST)

                    # Michelson contrast
                    if self.do_mcontrast:
                        mcontrast = ct.get_Michelson(profile)
                        self.show_plot.emit([], [], norm_profile, [], 0, 0)
                        self.show_mcontrast.emit(mcontrast)

                    # MTF
                    if self.do_mtf:
                        f_pos, mtf_norm, mtf50, mtf10, y_data_fitted = ct.get_MTF(norm_profile, PIX_SIZE)
                        if f_pos is not None:
                            self.show_plot.emit(f_pos, mtf_norm, norm_profile, y_data_fitted, mtf50, mtf10)
                            self.show_mtf.emit((mtf50, mtf10))
                        else:
                            self.show_plot.emit([], [], norm_profile, [], 0, 0)
                            self.show_mtf.emit((mtf50, mtf10))

                # Emit processed frame
                if self.proc_frame is not None:
                    self.img_processed.emit(self.proc_frame)

                self.update = False

    def enable_mtf(self, value:bool):
        """
        Enable or disable MTF constrast computation.

        Parameters
        ----------
        value : bool
        """
        self.do_mtf = value
        self.update = True

    def enable_mcontrast(self, value:bool):
        """
        Enable or disable Michelson contrast computation.

        Parameters
        ----------
        value : bool
        """
        self.do_mcontrast = value
        self.update = True

    @Slot(float, float, str)
    def update_roi(self, x:float, y:float, button_clicked:str):
        """
        Update ROI points based on mouse interaction.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        button_clicked : str
            'left' sets p1, anything else sets p2.
        """
        if button_clicked == 'left':
            self.p1 = (int(x), int(y))
        else:
            self.p2 = (int(x), int(y))
        self.update = True
        self.get_angle()

    def get_angle(self):
        """
        Compute and emit the angle between p1 and p2 relative to horizontal
        and vertical axes.
        """
        theta_rad = math.atan2(self.p2[1] - self.p1[1],
                               self.p2[0] - self.p1[0])
        theta_deg = math.degrees(theta_rad)
        vertical_angle   = abs(90 - abs(theta_deg))
        horizontal_angle = abs(theta_deg)
        self.show_angle.emit(horizontal_angle, vertical_angle)

    def stop(self):
        """
        Stop the processing thread and return the last ROI points.

        Returns
        -------
        tuple(tuple(int,int), tuple(int,int))
            The current ROI endpoints (p1, p2).
        """
        self.running = False
        self.wait()
        return (self.p1, self.p2)








# ====================================================================================
#                              Class ZoomableGraphicsView
# ====================================================================================
class ZoomableGraphicsView(QGraphicsView):
    """
    A QGraphicsView widget that supports smooth zooming and detects mouse
    clicks on images in the view. Emits image_clicked(x, y, button).
    """

    image_clicked = Signal(float, float, str)

    def __init__(self, parent: object = None):
        """
        Initialize the zoomable graphics view.

        Parameters
        ----------
        parent : object, optional
            Parent widget.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._zoom: int = 0
        self._mouse_press_pos = None 
        self._mouse_moved: bool = False
        self.do_mtf: bool = False
        self.do_mcontrast: bool = False
        self.do_grid: bool = False

    def reset_view(self) -> None:
        """Reset zoom to fit the entire scene."""
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 0

    def wheelEvent(self, event) -> None:
        """
        Handle mouse wheel zooming.

        Parameters
        ----------
        event : QWheelEvent
            Wheel event from the mouse.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8
        max_zoom = 10
        min_zoom = -10

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
            self._zoom += 1
        else:
            factor = zoom_out_factor
            self._zoom -= 1

        if self._zoom > max_zoom:
            self._zoom = max_zoom
            return
        if self._zoom < min_zoom:
            self._zoom = min_zoom
            return

        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        """
        Record the mouse press position to detect drag vs click.

        Parameters
        ----------
        event : QMouseEvent
            Mouse press event.
        """
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._mouse_press_pos = event.position()
            self._mouse_moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Track whether the mouse has moved enough to be considered a drag.

        Parameters
        ----------
        event : QMouseEvent
            Mouse move event.
        """
        if self._mouse_press_pos is not None:
            distance = (event.position() - self._mouse_press_pos).manhattanLength()
            if distance > QApplication.startDragDistance():
                self._mouse_moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """
        Detect image clicks and emit image_clicked(x, y, button).

        Parameters
        ----------
        event : QMouseEvent
            Mouse release event.
        """
        if self.do_mtf or self.do_mcontrast or self.do_grid:
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                if not self._mouse_moved:
                    scene_pos = self.mapToScene(event.position().toPoint())
                    for item in self.scene().items():
                        from PySide6.QtWidgets import QGraphicsPixmapItem
                        if isinstance(item, QGraphicsPixmapItem):
                            image_pos = item.mapFromScene(scene_pos)
                            if item.contains(image_pos):
                                if event.button() == Qt.LeftButton:
                                    self.image_clicked.emit(image_pos.x(), image_pos.y(), 'left')
                                elif event.button() == Qt.RightButton:
                                    self.image_clicked.emit(image_pos.x(), image_pos.y(), 'right')
                                break
        super().mouseReleaseEvent(event)










# ====================================================================================
#                                   Class MainWindow
# ====================================================================================
class MainWindow(QMainWindow):
    streaming = False
    exposure_input = None
    measurements = {}
    images = {}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Viewer")
        self.setGeometry(100, 100, 1800, 1200)

        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Controls
        self.device_selector = QComboBox()
        self.device_selector.setStyleSheet("color: cyan; font-weight: bold;")
        serials = list_cameras()
        print(serials)
        if len(serials)==0:
            print("ERROR: No cameras found. Program terminated.")
            sys.exit(-1)
        self.cameras = {}
        for serial, device in serials.items():
            self.cameras[serial] = Camera(serial, device,"")
            self.cameras[serial].connect()
            self.device_selector.addItem(serial, device)
            self.measurements[serial] = {"MTF50":0.0,"MTF10":0.0,"Michelson":0.0}
            self.images[serial] = {"MTF_raw":[],"MTF_proc":[],"Michelson_raw":[],"Michelson_proc":[]}
        self.device_selector.currentIndexChanged.connect(self.device_changed)

        self.exposure_input = QLineEdit("10000")
        self.exposure_input.setStyleSheet("color: cyan; font-weight: bold;")
        self.exposure_input.setPlaceholderText("Enter Exposure (e.g. 10000)")
        self.exposure_input.setFixedWidth(100)

        self.roi_angle_label = QLabel("ROI θ [H: 00.0 | V: 00.0]")
        self.roi_angle_label.setAlignment(Qt.AlignLeft)

        self.last_frame_time = None
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignLeft)

        # Top Control Panel
        top_panel = QHBoxLayout()
        top_panel.addWidget(QLabel("Camera:"))
        top_panel.addWidget(self.device_selector)
        top_panel.addWidget(QLabel("Exposure:"))
        top_panel.addWidget(self.exposure_input)
        top_panel.addStretch()
        top_panel.addWidget(self.roi_angle_label)
        top_panel.addWidget(self.fps_label)

        # Top panel style (slightly darker)
        top_widget = QWidget()
        top_widget.setLayout(top_panel)
        top_widget.setStyleSheet("background-color: #171717; color: white; padding: 6px;")

        # Image Viewer
        self.view = ZoomableGraphicsView()
        self.scene = self.view.scene()
        self.pixmap_item = self.scene.addPixmap(QPixmap())

        # FLOATING BUTTONS
            # Start/Stop button
        self.start_button = self.create_button("Start", "fa5s.play", checkable=True,
                                                    slot=self.toggle_start_stop,
                                                    object_name="start_button")

            # Fit button (fit the view widget to the size of the window)
        self.fit_button = self.create_button("Fit", "fa5s.expand", checkable=False,
                                                    slot=self.view.reset_view,
                                                    object_name="floating_button")

            # Grid button
        self.grid_button = self.create_button("Grid", "fa5s.th-large", checkable=True,
                                                    slot=self.toggle_grid,
                                                    object_name="check_button")

            # MTF contrast button
        self.mtf_button = self.create_button("MTF", "mdi.chart-bell-curve", checkable=True,
                                                    slot=self.mtf_toggled,
                                                    object_name="check_button")
            # Michelson contrast button
        self.michelson_button = self.create_button("Michelson", "fa5s.bars", checkable=True,
                                                    slot=self.mcontrast_toggled,
                                                    object_name="check_button")
        
        # Michelson contrast button
        self.turing_button = self.create_button("Turing", "mdi.texture", checkable=True,
                                                    slot=self.turing_toggled,
                                                    object_name="check_button")
        
            # Saturation indicator button (non-clickable)
        self.saturation_label = self.create_button("Saturated\n100%", "fa5s.exclamation-triangle", checkable=False,
                                                   icon_color="lightyellow", enabled=False,
                                                   object_name="saturation_label")
        
        if len(self.cameras) < 1:
            self.device_selector.setEnabled(False)
            self.start_button.setEnabled(False)

        # Right Panel
        self.right_panel_widget = QWidget()
        right_panel = QVBoxLayout(self.right_panel_widget)

        # Group box for the plot and labels
        plot_group = QGroupBox("Image Contrast")
        self.plot_group = plot_group
        plot_group_layout = QVBoxLayout()
        plot_group.setObjectName("plot_group")

        
        # MTF Group Box
        mtf_group = QGroupBox("Modulation Transfer Function (MTF) contrast")
        mtf_layout = QHBoxLayout()
        # Labels for MTF50 and MTF10
        self.mtf50_label = QLabel("0.00")
        self.mtf50_label.setObjectName("MtfLabel")
        self.mtf50_units_label = QLabel("MTF50\nCycles/mm")
        self.mtf50_units_label.setObjectName("Mtf_units_Label")
        self.mtf10_label = QLabel("0.00")
        self.mtf10_label.setObjectName("MtfLabel")
        self.mtf10_units_label = QLabel("MTF10\nCycles/mm")
        self.mtf10_units_label.setObjectName("Mtf_units_Label")
        # Add labels to the layout
        mtf_layout.addWidget(self.mtf50_label)
        mtf_layout.addWidget(self.mtf50_units_label)
        mtf_layout.addWidget(self.mtf10_label)
        mtf_layout.addWidget(self.mtf10_units_label)
        # Set layout to the group box
        mtf_group.setLayout(mtf_layout)
        # Add the group box to the main layout (for example, above the plot)
        plot_group_layout.addWidget(mtf_group)

        # Plot Group Boxvariant
        plot_widget_group = QGroupBox("ROI Plot")
        plot_widget_layout = QVBoxLayout()

        # Add the plot widget
        self.plot_widget = PlotWidget()
        self.plot_widget.setMinimumHeight(500)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_widget_layout.addWidget(self.plot_widget)
        # Set layout for the group box
        plot_widget_group.setLayout(plot_widget_layout)
        # Add the group box to the main layout
        plot_group_layout.addWidget(plot_widget_group, stretch=2)

        # Michelson Contrast Group Box
        contrast_group = QGroupBox("Michelson Contrast")
        contrast_layout = QHBoxLayout()
        # Labels for contrast
        self.mcontrast_label = QLabel("0.00")
        self.mcontrast_label.setObjectName("MtfLabel")
        self.mcontrast_units_label = QLabel("Normalized\nContrast (a.u.)")
        self.mcontrast_units_label.setObjectName("Mtf_units_Label")
        # Add labels to the layout
        contrast_layout.addWidget(self.mcontrast_label)
        contrast_layout.addWidget(self.mcontrast_units_label)
        contrast_layout.addStretch()
        # Set layout to the group box
        contrast_group.setLayout(contrast_layout)
        # Add the group box to the main plot group layout
        plot_group_layout.addWidget(contrast_group)

        # Set layout for the group box
        plot_group.setLayout(plot_group_layout)
        plot_group.setObjectName("PlotGroup")
        # Make sure the group box also expands
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Add group box to the right panel
        right_panel.addWidget(plot_group, stretch=2)
        # Add stretch at the bottom (optional)
        right_panel.addStretch()

        # View Container (for floating buttons)
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.view)
        view_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Attach floating buttons to view_container (so they overlay correctly)
        buttons = (self.start_button, self.mtf_button, self.michelson_button, 
                   self.grid_button, self.turing_button,self.fit_button, 
                   self.saturation_label)
        for btn in buttons:
            btn.setParent(view_container)

        # Position buttons manually
        start_x, start_y = 20, 20
        button_spacing, button_height = 10, 80

        for idx, btn in enumerate(buttons):
            btn.move(start_x, start_y + idx*(button_height + button_spacing))
            # Bring buttons to front and show
            btn.raise_()
            btn.show()
        self.saturation_label.move(start_x+100, start_y)
        self.saturation_label.setVisible(False)

        # Center layout: View + Right panel
        center_layout = QHBoxLayout()
        center_layout.addWidget(view_container, 7)
        center_layout.addWidget(self.right_panel_widget, 3)

        # Combine
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_widget)
        main_layout.addLayout(center_layout)
        main_widget.setLayout(main_layout)

        self.camera_thread = None
        self.streaming = False

        # Shared queue
        self.frame_queue = Queue(maxsize=5)
        self.roi_p1, self._roi_p2  = (1344//2,1344//2), ((1344//2)+35,(1344//2)+35)
        self.processor_thread = ProcessingThread(frame_queue=self.frame_queue)
        self.processor_thread.show_angle.connect(self.show_angle)
        self.processor_thread.set_roi_points(self.roi_p1, self._roi_p2)
        self.processor_thread.img_processed.connect(self.update_image)
        self.processor_thread.show_mtf.connect(self.show_mtf)
        self.processor_thread.show_mcontrast.connect(self.show_mcontrast)
        self.processor_thread.show_saturation.connect(self.show_saturation)
        self.processor_thread.show_plot.connect(self.plot_widget.update_plot)
        self.view.image_clicked.connect(self.processor_thread.update_roi)
        self.processor_thread.start()



    def create_button(
        self,
        text: str,
        icon_name: str,
        icon_color: str = "white",
        checkable: bool = True,
        slot=None,
        object_name: str = "",
        size: int = 80,
        icon_size: int = 32,
        toolbutton_style=Qt.ToolButtonTextUnderIcon,
        enabled = True,
        parent=None,
    ) -> QToolButton:
        btn = QToolButton(parent)
        btn.setText(text)
        btn.setToolButtonStyle(toolbutton_style)
        btn.setIcon(qta.icon(icon_name, color=icon_color))
        btn.setIconSize(QSize(icon_size, icon_size))
        btn.setFixedSize(size, size)
        btn.setCheckable(checkable)
        btn.setEnabled(enabled) 

        if slot is not None:
            # Auto-connect toggled or clicked depending on checkable
            if checkable:
                btn.toggled.connect(slot)
            else:
                btn.clicked.connect(slot)

        if object_name:
            btn.setObjectName(object_name)

        return btn





    @Slot(float, float)
    def show_angle(self, horizontal: float, vertical: float) -> None:
        """ Update the ROI angle label."""
        self.roi_angle_label.setText(f"ROI θ [H: {horizontal:3.1f} | V: {vertical:3.1f}]")


    @Slot(bool)
    def turing_toggled(self, checked: bool) -> None:
        """Toggle MTF contrast calculation mode."""
        if checked:
            self.michelson_button.setChecked(False)
            self.mtf_button.setChecked(False)
            self.grid_button.setChecked(False)
        self.processor_thread.do_turing = checked
        self.processor_thread.update = True



    @Slot(bool)
    def toggle_grid(self, checked: bool) -> None:
        """Toggle the grid overlay."""
        if checked:
            self.michelson_button.setChecked(False)
            self.mtf_button.setChecked(False)
            self.turing_button.setChecked(False)
        self.view.do_mtf = checked
        self.processor_thread.show_grid = checked
        self.processor_thread.update = True



    @Slot(bool)
    def mtf_toggled(self, checked: bool) -> None:
        """Toggle MTF contrast calculation mode."""
        self.reset_contrast_labels()
        self.processor_thread.do_mtf = checked
        if checked:
            self.michelson_button.setChecked(False)
            self.grid_button.setChecked(False)
            self.turing_button.setChecked(False)
        self.view.do_mtf = checked
        self.processor_thread.update = True
        

    @Slot(bool)
    def mcontrast_toggled(self, checked:bool) -> None:
        """Toggle Michelson contrast calculation mode."""
        self.reset_contrast_labels()
        self.processor_thread.do_mcontrast = checked
        if checked:
            self.mtf_button.setChecked(False)
            self.grid_button.setChecked(False)
            self.turing_button.setChecked(False)
        self.view.do_mcontrast = checked
        self.processor_thread.update = True


    @Slot(int)
    def device_changed(self, index: int) -> None:
        """Handle the selection of a new camera."""
        self.reset_contrast_labels()
        if self.streaming:
            self.stop_camera()
            self.start_camera()
        if self.exposure_input:
            self.exposure_input.setText(f"{self.cameras[self.device_selector.currentText()].get_exposure()}")


    def reset_contrast_labels(self):
        """Reset all contrast/MTF labels and clear the plot."""
        self.mtf50_label.setText("0.00")
        self.mtf10_label.setText("0.00")
        self.mcontrast_label.setText("0.00")
        self.plot_widget.reset_plot()
        self.show_contrast_panel()


    def show_contrast_panel(self) -> None:
        visible = self.mtf_button.isChecked() or self.michelson_button.isChecked()
        self.right_panel_widget.setVisible(visible)

        
    @Slot()
    def toggle_start_stop(self) -> None:
        """Start or stop streaming based on current state."""
        if self.streaming:
            self.stop_camera()
            # Change button to green for Start
            self.start_button.setText("Start")
            self.start_button.setIcon(qta.icon("fa5s.play", color="white"))
        else:
            self.start_camera()
            # Change button to red for Stop
            self.start_button.setText("Stop")
            self.start_button.setIcon(qta.icon("fa5s.stop", color="white"))


    def start_camera(self) -> None:
        """Start the camera streaming."""
        serial = self.device_selector.currentText()
        self.exposure_input.setText(f"{self.cameras[serial].get_exposure()}")
        exposure = self.exposure_input.text()
        exposure = float(exposure) if exposure else 10000
        # Camera Thread
        self.camera_thread = CameraThread(self.cameras[serial], frame_queue=self.frame_queue)
        self.camera_thread.frame_rate.connect(self.fps_label.setText)
        self.exposure_input.returnPressed.connect(self.on_exposure_changed)
        self.camera_thread.set_camera(self.cameras[serial])
        self.camera_thread.start_streaming()
        # Start the imaq thread
        self.camera_thread.start()
        self.streaming = True


    def stop_camera(self) -> None:
        """Stop the camera thread if active."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.streaming = False
        

    def on_exposure_changed(self) -> None:
        """Apply a new exposure value to the camera thread."""
        try:
            exposure_value = float(self.exposure_input.text())
        except ValueError:
            print("Invalid exposure value.")
            return
        if self.camera_thread:
            self.camera_thread.set_exposure(exposure_value)

    @Slot(np.ndarray)
    def show_mtf(self, mtf:np.ndarray) -> None:
        """
        Display MTF50 and MTF10 values.

        Parameters
        ----------
        mtf : (np.ndarray) Array containing MTF50 and MTF10 values.
        """
        if mtf[0]:
            self.mtf50_label.setText(f"{mtf[0]:3.2f}")
            self.mtf10_label.setText(f"{mtf[1]:3.2f}")
        else:
            self.mtf50_label.setText("0.00")
            self.mtf10_label.setText("0.00")


    @Slot(float)
    def show_mcontrast(self, mcontrast:float) -> None:
        """
        Display Michelson contrast.

        Parameters
        ----------
        mcontrast : (float) Michelson contrast value.
        """
        if mcontrast is not None:
            self.mcontrast_label.setText(f"{mcontrast:3.2f}")
        else:
            self.mcontrast_label.setText("0.00")


    @Slot(float)
    def show_saturation(self, saturation:float) -> None:
        """Display saturation percentage.'"""
        if saturation>0:
            self.saturation_label.setText(f"Saturated\n{saturation*100:3.2f}%")
            self.saturation_label.setVisible(True)
        else:
            self.saturation_label.setVisible(False)     


    @Slot(np.ndarray)
    def update_image(self, frame: np.ndarray) -> None:
        """
        Update the displayed image.

        Parameters
        ----------
        frame : (np.ndarray) Incoming BGR image frame.
        """
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.pixmap_item.setPixmap(pixmap)

    
    def show_error_message(self, icon, title_txt, error_txt, info_txt=None, detail_txt=None) -> None:
        """Display an error message dialog."""
        # Create and configure the QMessageBox
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title_txt)
        msg_box.setText(error_txt)
        if info_txt:
            msg_box.setInformativeText(info_txt)
        if detail_txt:
            msg_box.setDetailedText(detail_txt)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()


    def closeEvent(self, event) -> None:
        """Properly shut down camera and processing threads on window close."""
        self.stop_camera()
        self.camera_thread = None
        self.processor_thread.stop()
        self.processor_thread = None
        event.accept()










def load_stylesheet(filename: str) -> str:
    """Load QSS stylesheet from the same directory as the script."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, filename)

    with open(full_path, "r") as f:
        return f.read()
    

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New"))

    # Create a dark mode palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(37, 37, 38))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)

    # Stylesheet to match the dark mode palette
    app.setStyleSheet(load_stylesheet("dark_theme.qss"))

    window = MainWindow()
    window.show()

    if len(window.cameras) < 1:
        print("\033[91mCAMERA CONNECTION ERROR: No cameras were found. Please check if the camera drivers were loaded correctly.\033[0m")
        window.show_error_message(
            QMessageBox.Critical,
            "CAMERA CONNECTION ERROR",
            "No cameras were found. Please check if the camera drivers were loaded correctly."
        )
        window.close()
        sys.exit(1)

    sys.exit(app.exec())
