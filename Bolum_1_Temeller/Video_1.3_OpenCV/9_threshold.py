import cv2
import numpy as np

# Görüntüyü gri tonlamalı oku
img = cv2.imread('Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu.png', 0)

# 1. Global Thresholding (Binary)
# Tüm resim için eşik değeri 127 olarak sabitlenmiştir.
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. Adaptive Thresholding (Mean)
# Eşik değeri, pikselin (11x11) komşuluğunun ortalamasıdır.
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                            cv2.THRESH_BINARY, 11, 2)

# 3. Adaptive Thresholding (Gaussian)
# Eşik değeri, komşuluğun Gauss ağırlıklı toplamıdır.
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2)

# Sonuçları incele
cv2.imshow('Global', th1)
cv2.imshow('Adaptive Mean', th2)
cv2.imshow('Adaptive Gaussian', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()