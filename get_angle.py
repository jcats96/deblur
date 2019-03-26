import numpy as np
import cv2

img = cv2.imread('blurred.png',0)
(img_rows,img_cols) = img.shape
print('rows '+ str(img_rows))
print('cols '+ str(img_cols))

img_FFT = np.fft.fft2(img)
img_FFT = np.fft.fftshift(img_FFT)
img_abs = np.abs(img_FFT)
magnitude_spectrum = 10*np.log(np.abs(img_FFT))

#determine blur angle using Gabor filter
max_norm =0
max_angle=0
for angles in range(0,90,3):
    G = cv2.getGaborKernel((13,13),3,angles,1.25,1)
    G = cv2.normalize(G,G)
    temp_img = cv2.filter2D(img_abs,-1,G,cv2.CV_32F)
    temp_norm = np.linalg.norm(temp_img)
   # temp_img = 10 * np.log(np.abs(temp_img))
    cv2.imshow('temp' + str(angles), np.abs(temp_img).astype(np.uint8))

    print(angles)
    print(temp_norm)
    if temp_norm>max_norm:
        max_norm = temp_norm
        max_angle = angles
print('max response at ' + str(max_angle))

G = cv2.getGaborKernel((15,15),5,60,1.75,1)
G = G*255
cv2.imshow('gabor',np.abs(G).astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()
