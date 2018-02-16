import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input1.jpg')


def display_image(name, image):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Rotation matrix:
cos -sin
sin cos"""

height, width = image.shape[:2]  # store height,width of image, :2 means start 0 to 2, 2 excluding

rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

display_image("Original", image)
display_image("Rotation", rotated_image)