# ## Blob Detection

"""Blob: groups of connected pixels that share a common property"""

# Standard imports
import cv2
import numpy as np;

# Read image
image = cv2.imread("images/Sunflowers.jpg", cv2.IMREAD_GRAYSCALE)

# Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector() #for opencv2.0
is_cv3 = cv2.__version__.startswith("3.")
if is_cv3:
    detector = cv2.SimpleBlobDetector_create()
else:
    detector = cv2.SimpleBlobDetector()

# Detect blobs.
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
# the circle corresponds to the size of blob
blank = np.zeros((1, 1))	#creates a matrix of 1x1 with elements 0
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 255),
                          cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The function **cv2.drawKeypoints** takes the following arguments:
#
# **cv2.drawKeypoints**(input image, keypoints, blank_output_array, color, flags)
#
# flags:
# - cv2.DRAW_MATCHES_FLAGS_DEFAULT
# - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# - cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# - cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
