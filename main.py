#main
print('test')

import numpy as np
import cv2

# Load an image
img = cv2.imread('lena.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()