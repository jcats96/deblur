#main
print('test')

import numpy as np
import cv2
import math

# Load an image as greyscale
img = cv2.imread('lena.jpg',0)


(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))




#determine blur angle using Gabor filter

#im using degrees instead of radians and im not going to convert by hand because i dont have to
theta = math.radians(45)
#determine blur length using radial basis function network - train.py in other file?
blength = 10
#compute point spread function

PSF =  np.zeros((img_rows, img_cols, 1), dtype = "uint8")

#PSF[img_rows//2,img_cols//2] = 255/blength

# create a list of points with float coordinates along the length of theta
#arbitrary position starting at "0"
#theta = 0 deg is up
# theta = 90 deg is right
x_coords = [0]
y_coords = [0]
for  hyp in range(1,10):
    x_coords.append(hyp*math.sin(theta))
    y_coords.append(hyp*math.cos(theta))

print(x_coords)
print(y_coords)

#bilinearly interpolate, or something to place them in the PSF

for  index in range(0,10):
    point_row = (img_rows//2) + (x_coords[index])//1
    point_col = (img_cols // 2) + (y_coords[index]) // 1
    PSF[int(point_row),int(point_col)] += 255//blength

#reconstruct image using weiner filter


cv2.imshow('PSF',PSF)
cv2.waitKey(0)
cv2.destroyAllWindows()