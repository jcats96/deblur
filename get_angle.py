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
for angles in range(10,80,5):
    G = cv2.getGaborKernel((7,7),3,angles+90,1.5,1)
    G = cv2.normalize(G,G)
    temp_img = cv2.filter2D(magnitude_spectrum,-1,G,cv2.CV_32F)
    temp_norm = np.linalg.norm(temp_img)
   # temp_img = 10 * np.log(np.abs(temp_img))
    cv2.imshow('temp' + str(angles), cv2.convertScaleAbs(temp_img))

    print(angles)
    print(temp_norm)
    if temp_norm>max_norm:
        max_norm = temp_norm
        max_angle = angles
print('max response at ' + str(max_angle))

G = cv2.getGaborKernel((15,15),5,150,5,1)
G = G*255
cv2.imshow('gabor',np.abs(G).astype(np.uint8))
cv2.imwrite('gabor.png',np.abs(G).astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()
