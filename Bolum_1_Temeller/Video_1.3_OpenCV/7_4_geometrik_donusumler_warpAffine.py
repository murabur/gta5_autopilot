"""
================================================================================
🚨 REFACTOR (DÜZENLEME) BEKLEYEN MÜHENDİSLİK VE MATEMATİK HATALARI 🚨
Not: Bu kod blokları "çalışıyor" gibi görünse de otonom sürüş veya 
prodüksiyon seviyesi için aşağıdaki kusurları barındırmaktadır. 
Gelecekteki revizyonda düzeltilecektir:

1. DOSYA YOLU RİSKİ (Bug Potansiyeli):
   - Hata: "Klasör\Video" şeklindeki yollarda ters eğik çizgi (\) kullanılmış. 
     Python'da \v, \n, \t kaçış karakteridir (escape character). Klasör adı 
     "notlar" olsaydı \n yüzünden kod çökecekti.
   - Çözüm: Dosya yolunun başına 'r' (raw string) eklenmeli (r"Yol\...") 
     veya ileri eğik çizgi (/) kullanılmalı.

2. DÖNDÜRME (ROTATION) MANTIĞI YANILGISI: - Güncelleme: DÜZELTİLDİ.
   - Hata: Yorum satırındaki "Merkezi sol üst köşeye taşıyoruz" ifadesi 
     matematiksel olarak yanlıştır.
   - Doğrusu: getRotationMatrix2D, resmin orijinini (0,0) sol üste değil, 
     resmin merkezine (cx, cy) taşır, orada döndürür ve tekrar geri iterek 
     matrisi oluşturur.

3. EĞME (SHEAR) İŞLEMİNDE VERİ KAYBI (Kritik Mantık Hatası):
   - Hata: X ekseninde 0.3 oranında eğme (shear) yapıldığında, resmin 
     alt pikselleri sağa doğru kayar. Ancak warpAffine tuvali (cols, rows) 
     orijinal boyutta bırakıldığı için resmin sağ tarafı bıçak gibi kesilir.
   - Çözüm: Tuval dinamik büyütülmeli. new_cols = int(cols + (0.3 * rows)) 
     hesaplanıp tuval ölçüsü olarak bu verilmeli.

4. ÖLÇEKLEME (SCALE) İÇİN MÜHENDİSLİK EKSİKLİĞİ:
   - Hata: warpAffine ile resmi büyütmek pikselleri bozar (Nearest Neighbor).
   - Not Düşülecek: Otonom sürüş gibi hassas projelerde büyütme işlemi 
     matrisle değil, piksellerin arasını kalite kaybı olmadan dolduran 
     cv2.resize() ve Bicubic/Lanczos enterpolasyon algoritmalarıyla yapılır.

5. DOKÜMANTASYONDAKİ DİKKATSİZLİKLER:
   - İki farklı yere "5. ADIM" yazılmış.
   - Değişken açıklamaları ("cols kolondan gelir" vb.) gereğinden fazla 
     uzun. Sadece "Satır = Yükseklik (Y), Sütun = Genişlik (X)" 
     denilip geçilmeli.
================================================================================
"""



import cv2
import numpy as np


# Afin Dönüşüm Matrisi (2x3):
# [[a, b, tx],
#  [c, d, ty]]
#
# Kaydırma:  a=1, b=0, c=0, d=1, tx=X, ty=Y
# Döndürme:  cos(θ), -sin(θ), sin(θ), cos(θ)
# Eğme:      b=shear_x veya c=shear_y
# Ölçekleme: a=sx, d=sy
#
# Hepsi aynı matris yapısı — sadece sayılar değişiyor.
# Bu yüzden hepsinde cv2.warpAffine kullanıyoruz.


# 1. ADIM: Görüntüyü oku
img = cv2.imread("Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu.png")  #goruntu dosyası okunur.

rows, cols = img.shape[:2] #yükseklik, genişlik

# --- 2. ADIM: KAYDIRMA (Translation) ---
# M = [[1, 0, tx], [0, 1, ty]]
# tx: Yatay kayma (+ Sağ, - Sol) | ty: Dikey kayma (+ Aşağı, - Yukarı)
M_trans = np.float32([
    [1, 0, 100], # 100 piksel SAĞA
    [0, 1, 50]   # 50 piksel AŞAĞI
])

#img ifadesi ile görüntü birinci sırada verilir
#ikinci sırada görüntünün hangi matrise göre şekillendirileceği verilir.
# .imgshape bize yükseklik, genişlik yani rows, cols yani height, width verir.
#cv2.warpAffine görüntünün basılacağı tuval sınırlarını bizden cols, rows, yani genişlik, yükseklik yani width, height şeklinde ister.
res_trans = cv2.warpAffine(img, M_trans, (cols, rows)) 



# --- 3. ADIM: DÖNDÜRME (Rotation) ---
# getRotationMatrix2D(merkez, aci, olcek)

#merkez noktası ondalıklı sayı çıkmaması için // integer bölmesi. 
#ornek_goruntu.jpg rows = 720 , cols = 1280
#karıştırmamanız için cols columns, kolondan. Yanyana kaç tane kolon varsa genişlik odur. 1280*720 görüntüde yatayda 1280 tane kolon vardır.
#1280*720 görüntüde dikeyde 720 tane satır vardır.
#center 1280//2 = 640 , 720//2 = 360 center = (640,360)
#neden center alıyoruz? çünkü normalde matris çarpımı sol üst köşeden(0,0) noktasından döndürür. Biz 0,0 noktasını merkeze taşıyarak dönmenin buradan olmasını sağlıyoruz.
#dönmenin buradan olmasını sağlıyoruz. ardından cv2.getRotationMatrix2D'den gelen matris merkezi geri yerine kaydırıyor.
center = (cols // 2, rows // 2) 

#sırasıyla (640,360), 30 derece açı, büyütme faktörü x1
M_rot = cv2.getRotationMatrix2D(center, 30, 1.0) # 30 derece Sola (Saat yönünün tersine)
print("Dönüşüm matrisi\n", M_rot)

#sırasıyla img, cv2.getRotationMatrix2D(center, 30, 1.0), (1280,720)
res_rot = cv2.warpAffine(img, M_rot, (cols, rows))



# --- 4. ADIM: EĞME / BÜKME (Shear) ---
# M = [[1, shear_x, 0], [shear_y, 1, 0]]
M_shear = np.float32([
    [1, 0.3, 0], # X ekseninde eğme
    [0, 1,   0]
])
res_shear = cv2.warpAffine(img, M_shear, (cols, rows))




# 5. ADIM: Ölçekleme
# 1.0 = Orijinal boyut, 2.0 = 2 kat büyüme, 0.5 = yarı boyut
s_x = 1.5  # Genişliği %50 artır
s_y = 1.5  # Yüksekliği %50 artır

#
# Formül: [[s_x, 0, 0], [0, s_y, 0]]
M_scale = np.float32([
    [s_x, 0,   0],
    [0,   s_y, 0]
])

# Dikkat: Eğer büyütme yapıyorsan dsize'ı (tuval) da büyütmen gerekir, 
# yoksa resmin sığmayan kısımları kesilir.
res_scale = cv2.warpAffine(img, M_scale, (int(cols * s_x), int(rows * s_y)))




# --- 5. ADIM: AYRI PENCERELERDE GÖSTERİM ---
# Her pencere kendi ismiyle ve orijinal boyutunda açılır
cv2.imshow("1. Orijinal Goruntu", img)
cv2.imshow("2. Kaydirma (Translation)", res_trans)
cv2.imshow("3. Donme (Rotation)", res_rot)
cv2.imshow("4. Egme (Shear)", res_shear)
cv2.imshow("Olceklendirilmiş (Zoom)", res_scale)

# Pencereleri ekrana yaymak için (Opsiyonel: Üst üste binmemeleri için)
cv2.moveWindow("1. Orijinal Goruntu", 0, 0)
cv2.moveWindow("2. Kaydirma (Translation)", 400, 0)
cv2.moveWindow("3. Donme (Rotation)", 0, 400)
cv2.moveWindow("4. Egme (Shear)", 400, 400)

print("Kapatmak icin herhangi bir tusa basin...")
cv2.waitKey(0)
cv2.destroyAllWindows()