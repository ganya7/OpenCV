import cv2
import numpy as np


def display_image(name):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# create a black square
image = np.zeros((512, 512, 3), np.uint8)  # create a black rectangle
image_bw = np.zeros((512, 512), np.uint8)  # create a black rectangle without 3-dimension or for grayscaling
cv2.imshow("Color image", image)
cv2.imshow("Black image", image_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw line over the square
cv2.line(image, (0, 0), (100, 100), (255, 255, 0), 50)
display_image("Blue lines")

# draw rectangle over the square
cv2.rectangle(image, (100, 100), (200, 200), (255, 155, 55), 3)
display_image("Rectangle inside square")

# draw circle
cv2.circle(image, (100, 100), 50, (175, 114, 52), -1)  # -1 for fill color inside circle
display_image("Circle")

# draw polygon
# define 4 points
pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)
# reshape our points in form required by polylines
pts = pts.reshape((-1, 1, 2))  # reshape is function of numpy, used for polylines

cv2.polylines(image, [pts], True, (0, 0, 255), 3)  # True whether poly closed or not
display_image("Polygon")

# add text
cv2.putText(image, "Hello World!", (75, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 170, 0), 2)
display_image("Hello World")