from tkinter import Image
from turtle import width
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw

class PatternMaker:
    def __init__(self):
        pass  # No initialization needed for now

    def create_checkerboard(self, size=(800, 800), block_size=50):
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
    

    def MakeTuring(self, image:Image.Image, rep=20, radius=5, sharpen_percent=300):
        """
        Applies Turing pattern effect to an existing image.
        Args:
            image (PIL.Image): The input image.
            rep (int): Number of times to apply the blur and sharpen filters.
            radius (int): Radius for the blur and sharpen filters.
            sharpen_percent (int): Percent for the sharpen filter. """
        img = image
        for _ in range(rep):
            img = img.filter(ImageFilter.BoxBlur(radius=radius))
            img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen_percent, threshold=0))
        return img

    def CreateTuringPattern(self,size, rep=20, radius=5, sharpen_percent=300):
        """
        Creates a random Turing pattern image.
        Args:
            size (tuple): Size of the image in pixels (height, width).
            rep (int): Number of times to apply the blur and sharpen filters.
            radius (int): Radius for the blur and sharpen filters.
            sharpen_percent (int): Percent for the sharpen filter. """
        img = Image.fromarray((np.random.random(size)*255).astype(np.uint8))
        return self.MakeTuring(img, rep, radius, sharpen_percent)

    def CreateTuringImage(self, imaage:np.ndarray, rep=20, radius=5, sharpen_percent=300):
        if len(imaage.shape)==3:
            gray = cv2.cvtColor(imaage, cv2.COLOR_BGR2GRAY)
        else:
            gray = imaage
        # Apply Canny edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = 255 - cv2.Canny(blurred, 100, 200)
        img = Image.fromarray(edges)
        return self.MakeTuring(img, rep, radius, sharpen_percent)
    

    def DrawCircle(self, image, x_coord, y_coord, fill='Black', outline='white', circle_diameter=10, line_width=1):
        """
        Draws a circle on the given image.
        Args:
            image (PIL.Image): The image to draw on.
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


    def DrawRect(self, image, rect, fill='Black', outline='white', line_width=1):
        """
        Draws a rectangle on the given image.
        Args:
            image (PIL.Image): The image to draw on.
            rect (tuple): The rectangle defined as (left, top, right, bottom).
            fill (str): The fill color of the rectangle.
            outline (str): The outline color of the rectangle.
            line_width (int): The width of the outline.
        """
        draw = ImageDraw.Draw(image)
        draw.rectangle(rect, fill=fill, outline=outline, width=line_width)


    def DrawTriangle(self, image, rect, fill='Black', outline='white', line_width=1):
        """
        Draws a triangle on the given image.
        Args:
            image (PIL.Image): The image to draw on.
            rect (tuple): The bounding rectangle defined as (left, top, right, bottom).
            fill (str): The fill color of the triangle.
            outline (str): The outline color of the triangle.
            line_width (int): The width of the outline.
        """
        draw = ImageDraw.Draw(image)
        points = [  (rect[0],rect[3]),
                    (rect[0],rect[1]),
                    (rect[2],rect[1])]
        draw.polygon(points, fill=fill, outline=outline, width=line_width)
