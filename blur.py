
import numpy as np
import cv2
import math

# Load an image
img = cv2.imread('lenna_color.tif',0).astype('float32')


(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

#deg to rad
theta = 80
blength = 20

#compute point spread function

PSF =  np.zeros((img_rows, img_cols), float)

cv2.ellipse(PSF,(int(img_rows/2),int(img_cols/2)),(0,int(blength/2)),int(90-theta),0,360,1,2)
#normalize PSF
print(cv2.sumElems(PSF))
sum = cv2.sumElems(PSF)[0]
reciprocal = 1/sum
PSF = PSF*reciprocal
print('psf done')

PSF_F =  np.fft.fft2(PSF)


img_F = np.fft.fftshift(np.fft.fft2(img))

img_F_mag = 10*np.log(np.abs(img_F))
PSF_F_mag = 10*np.log(np.abs(PSF_F))

#
blur = cv2.filter2D(img,-1,PSF)

dst_F = img_F*PSF_F
dst_F_mag = 10*np.log(np.abs(dst_F))

dst = (np.fft.ifft2(np.fft.ifftshift(dst_F)))

dst = np.real(dst)

blur_F_mag = 10*np.log(np.abs(np.fft.fftshift(np.fft.fft2(blur))))


PSF = PSF *255*100
cv2.imshow('PSF',cv2.convertScaleAbs(PSF))
cv2.imshow('img',cv2.convertScaleAbs(img))
cv2.imshow('dst',cv2.convertScaleAbs(dst))
cv2.imshow('blur',cv2.convertScaleAbs(blur))
cv2.imshow('PSF_F',cv2.convertScaleAbs(PSF_F_mag))
cv2.imshow('img_F',cv2.convertScaleAbs(img_F_mag))
cv2.imshow('dst_F',cv2.convertScaleAbs(dst_F_mag))
cv2.imshow('blur_F',cv2.convertScaleAbs(blur_F_mag))

cv2.imwrite('lenna.png',img)
cv2.imwrite('blurred.png',blur)
cv2.imwrite('burredf.png',cv2.convertScaleAbs(blur_F_mag))

cv2.imwrite('PSF.png',PSF)
cv2.imwrite('PSF_f.png',cv2.convertScaleAbs(PSF_F_mag))
cv2.imwrite('dst_f.png',cv2.convertScaleAbs(dst_F_mag))



cv2.waitKey(0)
cv2.destroyAllWindows()

