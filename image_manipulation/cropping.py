import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input.jpg')


def display_image(name, image):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Cropping refers to extracting a segment of that image"""

# no direct function in opencv for cropping, but using np

height, width = image.shape[:2]

# Let's get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

# Let's get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

# Simply use indexing to crop out the rectangle we desire
cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.imshow("Cropped Image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()