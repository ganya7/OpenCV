import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input1.jpg')


def display_image(name, image):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Image pyramid different way of upscaling or downscaling, easily and quickly scale images
Scaling down reduces size by half and vice versa
Useful when making object detectors that scales images each time when it looks for an object"""

smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

cv2.imshow('Original', image)

cv2.imshow('Smaller ', smaller)
cv2.imshow('Larger ', larger)
cv2.waitKey(0)
cv2.destroyAllWindows()
