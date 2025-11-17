
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

    @staticmethod
    def slice_along_line(image: np.ndarray, p1: typing.Tuple[int, int], p2: typing.Tuple[int, int], thickness: int) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        """
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
         # Calculate the line length
        line_vec = p2_arr - p1_arr
        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            print("Warning: Points are identical. Returning None.")
            return None, None

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


    @staticmethod
    def slice_by_corners(image: np.ndarray, p1: typing.Tuple[int, int], p2: typing.Tuple[int, int], height: float) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts a rectangle from two opposite corners (p1, p2) with a given height perpendicular
        to the diagonal, assuming the rectangle may be rotated.

        Args:
            image (np.ndarray): Source image
            p1 (tuple): Top-left corner
            p2 (tuple): Bottom-right corner
            height (float): Height perpendicular to the diagonal (p1->p2)

        Returns:
            np.ndarray: Straightened rectangle slice
        """
        p1_arr = np.array(p1, dtype=np.float32)
        p2_arr = np.array(p2, dtype=np.float32)

        # Vector along the diagonal
        diag_vec = p2_arr - p1_arr
        diag_length = np.linalg.norm(diag_vec)
        if diag_length == 0:
            print("Warning: p1 and p2 are identical")
            return None

        # Unit vector along the diagonal
        diag_dir = diag_vec / diag_length

        # Perpendicular vector
        perp_dir = np.array([-diag_dir[1], diag_dir[0]])

        # To form the right triangles, scale the perpendicular vector
        # The perpendicular sides have length such that the rectangle height = given height
        # Offset formula derived from right triangle: the perpendicular side length along perp_dir
        half_height = height / 2.0

        # Compute the other two corners
        top_right = p1_arr + perp_dir * half_height * 2  # shifted from p1 along perpendicular
        bottom_left = p2_arr - perp_dir * half_height * 2  # shifted from p2 along negative perpendicular

        # Now we have all four corners in order: TL, TR, BR, BL
        src_pts = np.array([
            p1_arr,       # top-left
            top_right,    # top-right
            p2_arr,       # bottom-right
            bottom_left   # bottom-left
        ], dtype=np.float32)

        # Width is distance between top-left and top-right
        width = np.linalg.norm(top_right - p1_arr)

        # Height is distance between top-left and bottom-left
        height = np.linalg.norm(bottom_left - p1_arr)

        # Destination rectangle (straightened)
        dst_pts = np.array([
            [0, 0],
            [0, width],
            [height, width],
            [height, 0]
        ], dtype=np.float32)

        # Perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image, M, (int(height), int(width)))

        return result, src_pts


