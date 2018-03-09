# ## Mini Project # 6 - Car & Pedestrian Detection
#
# **NOTE**
# - If no video loads after running code, you may need to copy your *opencv_ffmpeg.dll*
# - From: *C:\opencv2413\opencv\sources\3rdparty\ffmpeg*
# - To: Where your python is installed e.g. *C:\Anaconda2\* \
# - Once it's copied you'll need to rename the file according to the version of OpenCV you're using.
# - e.g. if you're using OpenCV 2.4.13 then rename the file as:
# - **opencv_ffmpeg2413_64.dll** or opencv_ffmpeg2413.dll (if you're using an X86 machine)
# - **opencv_ffmpeg310_64.dll** or opencv_ffmpeg310.dll (if you're using an X86 machine)
#
# To find out where you python.exe is installed, just run these two lines of code:

# from cv2 import __version__
import sys

# print(__version__)
print(sys.version_info)

# ### Pedistrian Detection

import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('Haarcascades\haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('images/walking.avi')  # input 0 as parameter for reading input from camera

# Loop once video is successfully loaded
while cap.isOpened():

    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_LINEAR)  # decrease resolution of frame by 50%

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.2,
                                              minNeighbors=3)  # minNeighbors is sensitivity, reducing it more sensitive more false positives, increasing it less sensitive

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

# ### Car Detection
#

import cv2
import time
import numpy as np

# Create our body classifier
car_classifier = cv2.CascadeClassifier('Haarcascades\haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('images/cars.avi')

# Loop once video is successfully loaded
while cap.isOpened():

    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

# - **Full Body / Pedestrian Classifier ** - https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_fullbody.xml
# - **Car Classifier ** - http://www.codeforge.com/read/241845/cars3.xml__html
#
