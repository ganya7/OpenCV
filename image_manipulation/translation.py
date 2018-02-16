import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input1.jpg')


def display_image(name):
    cv2.imshow("" + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Transformation are geometric distortion enacted upon image
2 types affine and non-affine
affine: after transformatino parallelism maintained
non-affine: parrallelism not maintained,length,angle but preserve collinearity and incidence"""

"""Translation matrix:
1 0 Tx
0 1 Ty"""
height, width = image.shape[:2]  # store height,width of image, :2 means start 0 to 2, 2 excluding
quarter_height, quarter_width = height / 4, width / 4

# Tranlation matrix
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])  # float32 defines data type of numpy array used

image_translated = cv2.warpAffine(image, T, (width, height))
display_image("Original")
cv2.imshow("Translation", image_translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
