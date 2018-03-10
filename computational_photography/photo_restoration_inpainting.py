import cv2
import numpy as np

# load our damaged photo
image = cv2.imread('images/abraham.jpg')
cv2.imshow('Original damaged photo', image)
cv2.waitKey(0)

# load the photo where we have marked the damaged areas
marked_damages = cv2.imread('images/mask.jpg', 0)
cv2.imshow('Marked damages', marked_damages)
cv2.waitKey(0)

# lets make a mask out of our marked image be changing all colours that are not white to black
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold binary', thresh1)
cv2.waitKey(0)

# lets dilate(make thicker) the marks that we made since thresholding has narrowed it slightly
kernel = np.ones((7, 7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations=1)  # dilation adds px to the boundaries
cv2.imshow('Dilated mask', mask)
cv2.imwrite('images/abraham_mask.png', mask)

cv2.waitKey(0)
restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()

# inpainting is the process of reconstructing lost or deteriorated parts of images and videos. It is an advanced form of
# interpolation that can be used to replace lost or corrupted parts of the image data
#
# inpaint radius - radius of a circular neighbourhood of each point inpainted that is considered by the algorithm. Smaller values look less blurred while larger values look more pixelated or blurred
#
# inpaint methods
# INPAINT_NS - Navier-Stokes based method
# INPAINT_TELEA - method by Alexandru Telea - better as it integrates more seamlessly into the image
