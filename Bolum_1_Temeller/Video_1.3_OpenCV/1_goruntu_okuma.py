#terminale girilecek komutlar
#pip install opencv-python
#pip install numpy

import cv2                      #opencv
import numpy as np              #numpy


img = cv2.imread("Bolum_2_OpenCv\ornek_goruntu.png", cv2.IMREAD_COLOR) #dosya yolu, renk formatı

cv2.imshow("frame",img)         #pencere adı, cv2.imread'ın okuduğu dosyanın atandığı değişken

cv2.waitKey(0)                  #görüntüyü sayı ms kadar beklet. 0 = kapatılana kadar bekle
cv2.destroyAllWindows()         #bütün opencv pencerelerini kapat.