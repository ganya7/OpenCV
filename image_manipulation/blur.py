import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input.jpg')

"""Convolution is a mathematical operation performed on 2 functions producing a 3rd function
which is typically a modified version of 1 of the original functions
Output img = Image x Function(kernel size)"""

"""Blurring is an operation where we average the pixels within a region(kernel)"""
# kernel = 1/25 x matrix of 5x5 each element =1
# we multiply 1/25 to normalize ie sum =1 otherwise we would be increasing intensity

cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Creating our 3 x 3 kernel
kernel_3x3 = np.ones((3, 3), np.float32) / 9

# We use the cv2.fitler2D to conovlve the kernal with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred)
cv2.waitKey(0)

# Creating our 7 x 7 kernel
kernel_7x7 = np.ones((7, 7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Other commonly used blurring methods in OpenCV

# Averaging done by convolving the image with a normalized box filter.
# This takes the pixels under the box and replaces the central element
# Box size needs to odd and positive
blur = cv2.blur(image, (3, 3))
cv2.imshow('Averaging', blur)
cv2.waitKey(0)

# Instead of box filter, gaussian kernel
Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)

# Takes median of all the pixels under kernel area and central
# element is replaced with this median value
median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)

# Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

# box/average > median > gaussian

# Image De-noising - Non-Local Means Denoising
# Parameters, after None are - the filter strength 'h' (5-10 is a good range)
# Next is hForColorComponents, set as same value as h again
#
dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)

cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()

# There are 4 variations of Non-Local Means Denoising:
#
# cv2.fastNlMeansDenoising() - works with a single grayscale images
# cv2.fastNlMeansDenoisingColored() - works with a color image.
# cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
# cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
