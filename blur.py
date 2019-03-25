
import numpy as np
import cv2
import math

# Load an image
img = cv2.imread('lenna_color.tif',0).astype('float32')


(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

#deg to rad
theta = 60
blength = 30

#compute point spread function

PSF =  np.zeros((img_rows, img_cols, 1), float)

cv2.ellipse(PSF,(int(img_rows/2),int(img_cols/2)),(0,int(blength/2)),int(90-theta),0,360,1,2)
#normalize PSF
print(cv2.sumElems(PSF))
sum = cv2.sumElems(PSF)[0]
reciprocal = 1/sum
PSF = PSF*reciprocal
print('psf done')

blur = cv2.filter2D(img,-1,PSF)

#PSF_F =  np.fft.fft2(PSF)

img_F = np.fft.fftshift(np.fft.fft2(img))

#
img_F_mag = 10*np.log(np.abs(img_F))
#
#
#dst_F = img_F*PSF_F

##dst = np.fft.ifft2(np.fft.ifftshift(dst_F))

PSF = PSF *255*100
cv2.imshow('PSF',PSF.astype(np.uint8))
cv2.imshow('img',np.uint8(img))
#cv2.imshow('PSF_F',np.uint8(PSF_F))
cv2.imshow('img_F',np.uint8(img_F_mag))
cv2.imshow('filtered',np.uint8(blur))
cv2.imwrite('blurred.png',blur)
cv2.imwrite('PSF.png',PSF)


cv2.waitKey(0)
cv2.destroyAllWindows()

