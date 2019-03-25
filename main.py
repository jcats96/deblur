#main
print('test')
import numpy as np
import cv2
import math
# Load an image as greyscale
img = cv2.imread('lenna_color.tif',0)
(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

img_FFT = np.fft.fft2(img)


#determine blur angle using Gabor filter

theta = 90
#determine blur length using radial basis function network
blength = 20


#compute point spread function
PSF =  np.zeros((img_rows, img_cols, 1), float)
cv2.ellipse(PSF,(int(img_rows/2),int(img_cols/2)),(0,int(blength/2)),int(90-theta),0,360,1,2)
#normalize PSF
print(cv2.sumElems(PSF))
sum = cv2.sumElems(PSF)[0]
reciprocal = 1/sum
PSF = PSF*reciprocal

#create weiner filter

PSF_shifted = np.fft.fftshift(PSF)
PSF_fft = np.fft.fft2(PSF_shifted)
H_real = np.real(PSF_fft)

weiner_filter = np.conj(PSF_fft)/(cv2.pow(H_real,2)+0.0001)
#reconstruct image using weiner filter


#output_img = np.fft.ifft2(weiner_filter*img_FFT)


PSF = PSF *255*100
cv2.imshow('PSF',PSF.astype(np.uint8))
#.imshow('Output',output_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()