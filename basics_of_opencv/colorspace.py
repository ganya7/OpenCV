import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Arvind\Desktop\input1.jpg')
# opencv stored rgb as bgr due to how unsigned 32-bit integers are stored
# in memory, it still ends being stored as RGB.
# 0x00BBGGRR will be stored as 0x00RRGGBB
# individual color level of pixel
B, G, R = image[10, 542]  # show the rgb value of the pixel position
print(B, G, R)

# grayscale is 2 dimension
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
print(gray_image[58][105])

# HSV colorspace
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hue image", hsv_image)
cv2.imshow('Hue channel', hsv_image[:, :, 0])  # 0,1,2 are array postions
cv2.imshow('Saturation channel', hsv_image[:, :, 1])
cv2.imshow('Value channel', hsv_image[:, :, 2])
cv2.waitKey()
cv2.destroyAllWindows()

# individual channels in RGB
B, G, R = cv2.split(image)  # split function spits the image into each color index
cv2.imshow("Blue", B)
cv2.imshow("Green", G)
cv2.imshow("Red", R)
cv2.waitKey()
cv2.destroyAllWindows()

merged = cv2.merge([B, G, R])
cv2.imshow("Merged to recreate original", merged)
cv2.imshow("Original", image)
cv2.waitKey()
cv2.destroyAllWindows()

# amplify the blue colour
merged = cv2.merge([B + 100, G, R])
cv2.imshow("Blue amplified", merged)
cv2.waitKey()
cv2.destroyAllWindows()

# using numpy
# creating a matrix of zeroes with dimensions of image h x w
zeroes = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeroes, zeroes, R]))
cv2.imshow("Green", cv2.merge([zeroes, G, zeroes]))
cv2.imshow("Blue", cv2.merge([B, zeroes, zeroes]))
cv2.waitKey()
cv2.destroyAllWindows()