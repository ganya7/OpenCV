# ## Mini Project # 3 - Counting Circles and Ellipses

import cv2
import numpy as np

# Load image
image = cv2.imread("images/blobs.jpg", 0)
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Intialize the detector using the default parameters
is_cv3 = cv2.__version__.startswith("3.")
if is_cv3:
    detector = cv2.SimpleBlobDetector_create()
else:
    detector = cv2.SimpleBlobDetector()

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Display image with blob keypoints
cv2.imshow("Blobs using default parameters", blobs)
cv2.waitKey(0)

# Set our filtering parameters
# Initialize parameter settiing using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
# measure of being cicular , 1 : perfect circle, 0 : opposite
params.filterByCircularity = True
params.minCircularity = 0.9

# Set Convexity filtering parameters
# area of blob or area of convex hull
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters
# measure of ellipticalness (low being more elliptical, high being more circular)
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
is_cv3 = cv2.__version__.startswith("3.")
if is_cv3:
    detector = cv2.SimpleBlobDetector_create(params)
else:
    detector = cv2.SimpleBlobDetector(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# **NOTE** OpenCV 3.XX, use this line of code for intializing our blob detector
#
# `detector = cv2.SimpleBlobDetector_create(params)`
#
# OpenCV 2.4.X users use this:
#
# `detector = cv2.SimpleBlobDetector()`
