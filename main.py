#main
print('test')

import numpy as np
import cv2

# Load an image
img = cv2.imread('lena.jpg')

#determine blur angle using Gabor filter

#determine blur length

#compute point spread function

#reconstruct image using weiner filter


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()