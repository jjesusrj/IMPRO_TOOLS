
import numpy as np
import cv2
import typing

class ImageSlicer:
    """
    Utility class for extracting and straightening rectangular slices from an image.
    Contains methods for both angle-based and corner-based slicing.
    """
    def __init__(self):
        pass

    def slice_along_line(self, image: np.ndarray, p1: typing.Tuple[int, int], p2: typing.Tuple[int, int], thickness: int) -> typing.Optional[np.ndarray]:
        """
        [Original Function, Renamed for Clarity]
        Extracts a rectangular slice defined by a centerline (p1 to p2) and a thickness,
        and straightens it using a perspective transform. This is useful for extracting
        rotated or angled regions.

        Args:
            image (np.ndarray): The source image.
            p1 (tuple): (x, y) coordinate for the start of the line (center).
            p2 (tuple): (x, y) coordinate for the end of the line (center).
            thickness (int): Thickness of the slice in pixels.

        Returns:
            np.ndarray: A new image containing the straightened slice, or None if points are identical.
        """
        p1_arr, p2_arr = np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)
        line_vec = p2_arr - p1_arr
        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            print("Warning: Points are identical. Returning None.")
            return None

        # Normalize the direction vector
        direction = line_vec / line_length
        # Perpendicular vector (for defining the slice width)
        perp_direction = np.array([-direction[1], direction[0]])

        # Calculate corners of the rectangle
        offset = (thickness / 2.0) * perp_direction
        src_pts = np.array([
            p1_arr - offset,
            p1_arr + offset,
            p2_arr + offset,
            p2_arr - offset
        ], dtype=np.float32)

        # Destination rectangle (straightened)
        dst_pts = np.array([
            [0, 0],
            [0, thickness],
            [line_length, thickness],
            [line_length, 0]
        ], dtype=np.float32)

        # Perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image, M, (int(line_length), thickness))

        return result, src_pts

    def slice_by_corners(self, image: np.ndarray, p1: typing.Tuple[int, int], p2: typing.Tuple[int, int]) -> typing.Optional[np.ndarray]:
        """
        [New Function]
        Extracts an axis-aligned rectangular slice defined by two opposite corners (p1 and p2).
        It is robust to the order of p1 and p2 (i.e., it doesn't matter if p1 is top-left or bottom-right).

        Args:
            image (np.ndarray): The source image.
            p1 (tuple): (x, y) coordinate of the first corner.
            p2 (tuple): (x, y) coordinate of the second, opposite corner.

        Returns:
            np.ndarray: A new image containing the sliced rectangle, or None if the area is zero.
        """
        x1, y1 = p1
        x2, y2 = p2

        # 1. Determine True Bounds (Min/Max X and Y)
        # This is the key step to making the order of p1 and p2 irrelevant.
        min_x = int(min(x1, x2))
        max_x = int(max(x1, x2))
        min_y = int(min(y1, y2))
        max_y = int(max(y1, y2))

        # Calculate final width and height
        width = max_x - min_x
        height = max_y - min_y

        # Check for valid area
        if width <= 0 or height <= 0:
            print("Warning: Area is zero or negative. Returning None.")
            return None

        # 2. Define Source Points (The four corners of the extracted box)
        # We define them in a consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        src_pts = np.array([
            [min_x, min_y],  # Top-Left (TL)
            [max_x, min_y],  # Top-Right (TR)
            [max_x, max_y],  # Bottom-Right (BR)
            [min_x, max_y]   # Bottom-Left (BL)
        ], dtype=np.float32)

        # 3. Define Destination Points (The unrotated output image)
        dst_pts = np.array([
            [0, 0],          # TL maps to (0, 0)
            [width, 0],      # TR maps to (width, 0)
            [width, height], # BR maps to (width, height)
            [0, height]      # BL maps to (0, height)
        ], dtype=np.float32)

        # 4. Apply Transform
        # A simple crop (slicing) would also work here, but using the perspective
        # transform method keeps the approach consistent with the 'slice_along_line' method.
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image, M, (width, height))

        return result
