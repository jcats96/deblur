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

#determine blur angle using Gabor filter
max_norm =0
max_angle=0
for angles in range(180):
    G = cv2.getGaborKernel((11,11),3,angles,2,1)
    temp_img = cv2.filter2D(magnitude_spectrum,-1,G)
    temp_norm = cv2.norm(temp_img)
    print(temp_norm)
    if temp_norm>max_norm:
        max_norm = temp_norm
        max_angle = angles
print('max response at ' + str(max_angle))



theta =   60
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
k = 0.001
weiner_filter = np.divide(np.conj(PSF_fft),H_squared+k)
#reconstruct image using weiner filter
print('weiner filter shape')
print(weiner_filter.shape)

weiner_filter = np.fft.fftshift(weiner_filter)

output_img_fft = img_FFT.copy()

for rows in range(img_rows):
    for cols in range(img_cols):
        output_img_fft[rows,cols] =weiner_filter[rows,cols]*img_FFT[rows,cols]


output_img_fft = np.fft.ifftshift(output_img_fft)
output_img = np.fft.ifft2(output_img_fft)

output_img = np.fft.fftshift(output_img)

output_img = np.real(output_img)

print(output_img.shape)
print(output_img.dtype)

mag_out = 10*np.log(np.abs(output_img_fft))

mag_PSF = 10*np.log(np.abs(PSF_fft))

cv2.imshow('img',img.astype(np.uint8))

G = cv2.getGaborKernel((11, 11), 3, max_angle, 2, 1)
temp_img = cv2.filter2D(magnitude_spectrum, -1, G)
temp_img = 10*np.log(np.abs(temp_img))


cv2.imshow('Gabor',temp_img.astype(np.uint8))

cv2.imshow('img_fft',magnitude_spectrum.astype(np.uint8))
cv2.imshow('Output FFT',mag_out.astype(np.uint8))
#cv2.imshow('magpsf',mag_PSF.astype(np.uint8))
cv2.imshow('Output',np.abs(output_img).astype(np.uint8))
#cv2.imwrite('Output.png',output_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()