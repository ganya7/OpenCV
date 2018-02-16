import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input1.jpg')


def display_image(name, image):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Interpolation is a method of constructing new data points
within the range of a discrete set of known data points """

# cv2.INTER_AREA:    Good for shrinking or down sampling
# cv2.INTER_NEAREST: Fastest
# cv2.INTER_LINEAR:  Good for zooming or up sampling (default)
# cv2.INTER_CUBIC:   Better
# cv2.INTER_LANCZOS4:    Best

# make image 3/4th of original size
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
cv2.imshow("Scaling - Linear", image_scaled)
cv2.waitKey(0)

image_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Scaling - Cubic", image_scaled)
cv2.waitKey(0)

# skew the resizing by setting the exact dimension, dsize = output size
image_scaled = cv2.resize(image, (900, 400), interpolation=cv2.INTER_AREA)
cv2.imshow("Scaling - Area", image_scaled)
cv2.waitKey(0)

cv2.destroyAllWindows()