import cv2

img = cv2.imread(r"Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu.png")

# Saat yönünde 90 derece
res = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #sadece 90 - 180 - 270 için çalışır ara açılarda warpAffine gereklidir.
#Çözünürlük Değişimi: 90 veya 270 derece döndürme yaptığında görüntünün genişlik ve yükseklik değerleri yer değiştirir. 
#(Örn: 1280x720 iken 720x1280 olur). 
# warpAffine kullanımında tuval boyutu (dsize) elle ayarlanmazsa resim kesilir, ama cv2.rotate bunu otomatik halleder.

cv2.imshow("90 Derece", res)
cv2.waitKey(0)