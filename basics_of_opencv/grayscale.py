import cv2

image = cv2.imread(r'C:\Users\Arvind\Desktop\input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
gray_image2 = cv2.imread(r'C:\Users\Arvind\Desktop\input.jpg',0)  # prcoess to convert to grayscale same but done at time of reading itsef
cv2.imshow('Grayscale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()