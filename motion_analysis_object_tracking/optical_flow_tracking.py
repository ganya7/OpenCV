##Optical flow object tracking

# Lucas Kanade Optical flow in openCV

import cv2
import numpy as np

# Load video stream
cap = cv2.VideoCapture('images/walking.avi')

# set parameter for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# set parameters for lucas kanade optical flow
lucas_kanade_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# create some random colours
# used to create our trails for object movement in the image
color = np.random.randint(0, 255, (100, 3))

# take first frame and find corners in it
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# find initial corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# create a mask image for drawing purpose
mask = np.zeros_like(prev_frame)

while (1):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_corners, None,
                                                           **lucas_kanade_params)

    # select and store good points
    good_new = new_corners[status == 1]
    good_old = prev_corners[status == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    # show optical flow
    cv2.imshow('Optical Flow Lucas Kanade', img)
    if cv2.waitKey(1) == 13:  # 13 is the enter key
        break

    # now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prev_corners = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

# Dense optical flow

import cv2
import numpy as np

# load video stream
cap = cv2.VideoCapture('images/walking.avi')

# get the first frame
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:
    # read of video file
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # computes the dense optical flow using the Gunnar Farneback's algorithm
    flow = cv2.calcOpticalFlowFarneback(previous_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # use flow to calculate the magnitude (speed) and angle of motion
    # use these values to calculate the color to reflect speed and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * (180 / (np.pi / 2))
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # show our demo of Dense Optical Flow
    cv2.imshow('Dense Optical Flow', final)
    if cv2.waitKey(1) == 13:  # 13 is Enter key
        break

    # store current image as previous image
    previous_gray = next

cap.release()
cv2.destroyAllWindows()

# color represents movement, intensity represents speed of movement
