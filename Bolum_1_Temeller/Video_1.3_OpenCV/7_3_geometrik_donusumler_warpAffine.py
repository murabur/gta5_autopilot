import cv2
import numpy as np

# 1. ADIM: Görüntüyü oku
img = cv2.imread("Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu.png")
rows, cols = img.shape[:2]

# --- 2. ADIM: KAYDIRMA (Translation) ---
# M = [[1, 0, tx], [0, 1, ty]]
# tx: Yatay kayma (+ Sağ, - Sol) | ty: Dikey kayma (+ Aşağı, - Yukarı)
M_trans = np.float32([
    [1, 0, 100], # 100 piksel SAĞA
    [0, 1, 50]   # 50 piksel AŞAĞI
])
res_trans = cv2.warpAffine(img, M_trans, (cols, rows))

# --- 3. ADIM: DÖNDÜRME (Rotation) ---
# getRotationMatrix2D(merkez, aci, olcek)
center = (cols // 2, rows // 2)
M_rot = cv2.getRotationMatrix2D(center, 30, 1.0) # 30 derece Sola (Counter-clockwise)
res_rot = cv2.warpAffine(img, M_rot, (cols, rows))

# --- 4. ADIM: EĞME / BÜKME (Shear) ---
# M = [[1, shear_x, 0], [shear_y, 1, 0]]
M_shear = np.float32([
    [1, 0.3, 0], # X ekseninde eğme
    [0, 1,   0]
])
res_shear = cv2.warpAffine(img, M_shear, (cols, rows))

# --- 5. ADIM: AYRI PENCERELERDE GÖSTERİM ---
# Her pencere kendi ismiyle ve orijinal boyutunda açılır
cv2.imshow("1. Orijinal Goruntu", img)
cv2.imshow("2. Kaydirma (Translation)", res_trans)
cv2.imshow("3. Donme (Rotation)", res_rot)
cv2.imshow("4. Egme (Shear)", res_shear)

# Pencereleri ekrana yaymak için (Opsiyonel: Üst üste binmemeleri için)
cv2.moveWindow("1. Orijinal Goruntu", 0, 0)
cv2.moveWindow("2. Kaydirma (Translation)", 400, 0)
cv2.moveWindow("3. Donme (Rotation)", 0, 400)
cv2.moveWindow("4. Egme (Shear)", 400, 400)

print("Kapatmak icin herhangi bir tusa basin...")
cv2.waitKey(0)
cv2.destroyAllWindows()