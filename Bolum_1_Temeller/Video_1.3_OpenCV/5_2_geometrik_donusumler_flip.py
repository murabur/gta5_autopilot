import cv2
import numpy as np #gerekli değil(gerekli) el alışkanlığı

# 1. ADIM: Görüntüyü oku
img = cv2.imread("Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu.png")

# --- 2. ADIM: AYNALAMA (Flip) ---
# flipCode = 1  : Yatay (Horizontal) -> Sağ-sol takas edilir.
# flipCode = 0  : Dikey (Vertical)   -> Üst-alt takas edilir (Takla atar).
# flipCode = -1 : Her iki eksen      -> Hem takla atar hem yön değiştirir.

horizontal_flip = cv2.flip(src=img, flipCode=1)  # Yolun gidiş yönünü değiştirir
vertical_flip   = cv2.flip(src=img, flipCode=0)  # Görüntüyü ters çevirir
both_flip       = cv2.flip(src=img, flipCode=-1) # 180 derece döndürme etkisi yapar

# --- 3. ADIM: GÖSTERİM ---
cv2.imshow("Orijinal", img)
cv2.imshow("Yatay Aynalama (1)", horizontal_flip)
cv2.imshow("Dikey Aynalama (0)", vertical_flip)
cv2.imshow("Cift Eksen (-1)", both_flip)

cv2.waitKey(0)
cv2.destroyAllWindows()