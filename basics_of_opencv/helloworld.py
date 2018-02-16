import cv2
import numpy as np
print(cv2.__version__)
input = cv2.imread(r'C:\Users\Arvind\Desktop\input.jpg')
cv2.imshow('Hello World',input)	#helloworld is the title of the window
cv2.waitKey(0)	#delay in ms, 0 infinite or keyboard input
print(input.shape)	#prints the dimension of the image and the color system, shape is defined in cv2
cv2.destroyAllWindows()	#without this statement the program would crash