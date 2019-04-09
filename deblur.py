#main
print('test')
import numpy as np
import cv2
import math
# Load an image as greyscale
img = cv2.imread('blurred.png',0)
(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

img_FFT = np.fft.fft2(img)
img_FFT = np.fft.fftshift(img_FFT)
magnitude_spectrum = 10*np.log(np.abs(img_FFT))


theta =   20
#determine blur length using radial basis function network
blength = 30


#compute point spread function
PSF =  np.zeros((img_rows, img_cols), float)

cv2.ellipse(PSF,(int(img_rows/2),int(img_cols/2)),(0,int(blength/2)),int(90-theta),0,360,255,2)
#normalize PSF
print(cv2.sumElems(PSF))
sum = cv2.sumElems(PSF)[0]
reciprocal = 1/sum
PSF = PSF*reciprocal
PSF = PSF *1
print('psf done')
print(cv2.sumElems(PSF)[0])
#PSF = PSF *255
#create weiner filter

PSF_fft = np.fft.fft2(PSF)
H_squared = np.power(PSF_fft,2)
k = 0.0005
weiner_filter = np.divide(np.conj(PSF_fft),H_squared+k)
#reconstruct image using weiner filter
print('weiner filter shape')
print(weiner_filter.shape)

weiner_filter = np.fft.fftshift(weiner_filter)

output_img_fft = img_FFT.copy()

for rows in range(img_rows):
    for cols in range(img_cols):
        output_img_fft[rows,cols] =weiner_filter[rows,cols]*img_FFT[rows,cols]


output_img_fft_shifted = np.fft.ifftshift(output_img_fft)
output_img = np.fft.ifft2(output_img_fft_shifted)

output_img = np.fft.fftshift(output_img)

output_img = np.real(output_img)

print(output_img.shape)
print(output_img.dtype)

mag_out = 10*np.log(np.abs(output_img_fft))

mag_PSF = 10*np.log(np.abs(PSF_fft))

mag_W = 10*np.log(np.abs(weiner_filter))

cv2.imshow('img',cv2.convertScaleAbs(img))


cv2.imshow('img_fft',cv2.convertScaleAbs(magnitude_spectrum))
cv2.imshow('Weiner FFT',cv2.convertScaleAbs(mag_W))
cv2.imwrite('weinerfft.png',cv2.convertScaleAbs(mag_W))
cv2.imshow('Output FFT',cv2.convertScaleAbs(mag_out))
cv2.imwrite('output_fft.png',cv2.convertScaleAbs(mag_out))
#cv2.imshow('magpsf',mag_PSF.astype(np.uint8))
cv2.imshow('Output',cv2.convertScaleAbs(output_img))
#cv2.imwrite('Output.png',output_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()