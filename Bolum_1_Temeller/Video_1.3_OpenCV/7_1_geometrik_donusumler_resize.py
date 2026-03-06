import cv2
import numpy as np

# 1. ADIM: Görüntüyü oku
img = cv2.imread("Bolum_1_Temeller/Video_1.3_OpenCV/ornek_goruntu.png")

# --- 2. ADIM: KÜÇÜLTME (Downscaling) ---
small_img = cv2.resize(
    src=img, 
    dsize=(640, 480),           # Hedef Boyut: (Genişlik, Yükseklik) height - width 
    interpolation=cv2.INTER_AREA # Küçültme için uygun
)

# --- 3. ADIM: BÜYÜTME (Upscaling) ---
# Uzaktaki bir trafik tabelasını "zoom" yapıp okumak istediğinde kullanılır.
# INTER_CUBIC: 4x4 çevre piksellerine bakarak en keskin büyütmeyi yapar (Yavaş ama kaliteli).
large_img = cv2.resize(
    src=img, 
    dsize=None,                 # dsize None ise fx ve fy kullanılır
    fx=1.5,                     # Genişliği 1.5 kat artır
    fy=1.5,                     # Yüksekliği 1.5 kat artır
    interpolation=cv2.INTER_CUBIC # Büyütme için uygun
)

# --- 4. ADIM: CANLI GÖSTERİM (OpenCV Window) ---
cv2.imshow("Orijinal Goruntu", img)
cv2.imshow("Kucultulmus (INTER_AREA)", small_img)
cv2.imshow("Buyutulmus (INTER_CUBIC)", large_img)

# Pencerenin kapanması için bir tuşa basılmasını bekle
cv2.waitKey(0)
cv2.destroyAllWindows()