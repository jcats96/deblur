
import numpy as np
import cv2
import math

# Load an image
img = cv2.imread('lena.jpg',0).astype('float32')


(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

#deg to rad
theta = math.radians(45)
blength = 6

#compute point spread function

PSF =  np.zeros((img_rows, img_cols, 1),   dtype = "uint8")

# create a list of points with float coordinates along the length of theta
#arbitrary position starting at "0"
#theta = 0 deg is up
# theta = 90 deg is right
x_coords = [0]
y_coords = [0]
for  hyp in range(1,blength):
    x_coords.append(hyp*math.sin(theta))
    y_coords.append(hyp*math.cos(theta))

print(x_coords)
print(y_coords)

#bilinearly interpolate, or something to place them in the PSF

#for now just round to the nearest neighbor?
for  index in range(0,blength):
    point_row = (img_rows//2) + (x_coords[index])//1
    point_col = (img_cols // 2) + (y_coords[index]) // 1
    PSF[int(point_row), int(point_col)] += 255# // blength


# use a kernel



#or use an FFT?


PSF_F =  (np.fft.fft2(PSF))

img_F = np.fft.fftshift(np.fft.fft2(img))

#
img_F_mag = 10*np.log(np.abs(img_F))

PSF_F_mag =  10*np.log(np.abs(PSF_F))

dst_F = img_F*PSF_F

dst = np.fft.ifft2(np.fft.ifftshift(dst_F))

cv2.imshow('PSF',PSF)
cv2.imshow('img',np.uint8(img))
cv2.imshow('PSF_F',np.uint8(PSF_F_mag))
cv2.imshow('img_F',np.uint8(img_F_mag))


cv2.waitKey(0)
cv2.destroyAllWindows()


