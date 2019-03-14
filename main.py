#main
print('test')

import numpy as np
import cv2

# Load an image
img = cv2.imread('lena.jpg')

(img_rows,img_cols,img_channels) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

#determine blur angle using Gabor filter
theta = 0
#determine blur length using radial basis function network - train.py in other file?
blength = 10
#compute point spread function

PSF =  np.zeros((img_rows, img_cols, 1), dtype = "uint8")

PSF[img_rows//2,img_cols//2] = 255/blength

# create a list of points with float coordinates along the length of theta

#bilinearly interpolate to place them in the PSF



#reconstruct image using weiner filter


cv2.imshow('PSF',PSF)
cv2.waitKey(0)
cv2.destroyAllWindows()