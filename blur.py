
import numpy as np
import cv2
import math


def calcPSF(filter_rows,filter_cols, len,theta):
    h = np.zeros((img_rows, img_cols, 1),float)
    cv2.ellipse(h, (filter_rows // 2, filter_cols // 2), (0,len // 2), 90 - theta, 0, 360, 255, cv2.FILLED)

    return h




# Load an image
img = cv2.imread('lenna_color.tif',0).astype('float32')


(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

#deg to rad
theta = 45
blength = 30

#compute point spread function

#PSF = calcPSF(img_rows,img_cols,blength,theta)

#PSF = PSF.astype(float)

# use a kernel

blursize = 10

kern = np.zeros((blursize, blursize, 1),float)
for i in range(blursize):
    kern[i,i] = 1/blursize


blur = cv2.filter2D(img,-1,kern)

#or use an FFT?


#PSF_F =  np.fft.fft2(PSF)

#img_F = np.fft.fftshift(np.fft.fft2(img))

#
#img_F_mag = 10*np.log(np.abs(img_F))
#
#
#dst_F = img_F*PSF_F

##dst = np.fft.ifft2(np.fft.ifftshift(dst_F))

#cv2.imshow('PSF',np.uint8(PSF))
cv2.imshow('img',np.uint8(img))
#cv2.imshow('PSF_F',np.uint8(PSF_F))
#cv2.imshow('img_F',np.uint8(img_F_mag))
cv2.imshow('filtered',np.uint8(blur))
cv2.imwrite('blurred.png',blur)



cv2.waitKey(0)
cv2.destroyAllWindows()

