"""Contours are continuous lines or curves that bound or cover the full boundary of an object in image"""

# # Contours

import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('images/shapes_donut.jpg')
cv2.imshow('Input Image', image)
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)

# Finding Contours
# Use a copy of your image e.g. edged.copy(), since findContours alters the image
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #opencv2.x
_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)
print(contours)
print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)  # -1 indicates draw all contours or use specified index

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# **cv2.findContours(image, Retrieval Mode, Approximation Method)**
#
# Returns -> contours, hierarchy
#
# **NOTE** In OpenCV 3.X, findContours returns a 3rd argument which is ret (or a boolean indicating if the function was successfully run).
#
# If you're using OpenCV 3.X replace line 12 with:
#
# _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# The variable 'contours' are stored as a numpy array of (x,y) points that form the contour
#
# While, 'hierarchy' describes the child-parent relationships between contours (i.e. contours within contours)
#
#
#
# #### Approximation Methods
#
# Using cv2.CHAIN_APPROX_NONE stores all the boundary points. But we don't necessarily need all bounding points. If the points form a straight line, we only need the start and ending points of that line.
#
# Using cv2.CHAIN_APPROX_SIMPLE instead only provides these start and end points of bounding contours, thus resulting in much more efficent storage of contour information..
# cv2.
# cv2.RETR_EXTERNAL retrieves only outer or external contours only
# cv2.RETR_LIST retrieves all contours
# cv2.RETR_CCOMP retrieves all in a 2 level hierarchy
# cv2.RETR_TREE retrieves all in full hierarchy
