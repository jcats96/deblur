#main
print('test')

import numpy as np
import cv2

# Load an image
img = cv2.imread('lena.jpg')



#determine blur angle using Gabor filter
theta = 0
#determine blur length using radial basis function network - train.py in other file?
blength = 2
#compute point spread function

PSF =  np.zeros((256, 256, 1), dtype = "uint8")



#reconstruct image using weiner filter


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()