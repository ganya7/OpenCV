##Object tracking
from audioop import max
import cv2
import numpy as np

# initialize camera
cap = cv2.VideoCapture(0)

# define range of blue coloyr in HSV
lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])

# create empty points array
points = []

# get default camera window size
ret, frame = cap.read()
height, width = frame.shape[:2]
frame_count = 0

while True:

    # capture webcam frame
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    # Find contours
    img, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create empty centre array to store centroid of mass
    # initialization of centre of array
    center = int(height / 2), int(width / 2)

    if len(contours) > 0:

        # get largest contour and its center
        # c = max(contours, key=cv2.contourArea)
        c_list = sorted(contours, key=cv2.contourArea, reverse=True)
        c = c_list[0]
        (x, y), radius = cv2.minEnclosingCircle(c)  # previously usd bounding rectangles in contours
        M = cv2.moments(c)  # trail is created by logging the moments
        try:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        except:
            center = int(height / 2), int(width / 2)

        # allow only contours that have radius larger than 25 px
        if radius > 25:  # threshold for the size of the object, if under 25px stop tracking
            # draw circle and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    # log center points
    points.append(center)

    # loop over the set of tracked points
    if radius > 25:
        for i in range(1, len(points)):
            try:
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)  # plot the trail points
            except:
                pass

            # make frame count 0
            frame_count = 0
    else:
        # count frames
        frame_count += 1

        # if we count 10 frames without object lets delete our trail
        # this prevents our trail from becoming too big or exploding
        if frame_count == 10:
            points = []
            # when frame_count reaches 20 lets clear our trail
            frame_count = 0

    # display our object tracker
    frame = cv2.flip(frame, 1)
    cv2.imshow("Object tracker", frame)

    if cv2.waitKey(1) == 13:  # 13 is Enter key
        break

cap.release()
cv2.destroyAllWindows()
