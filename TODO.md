YOLO
- Model yeni veri setiyle eğitilecek ve paylaşılacak
Kontrol/Algı
- Ekrandaki en büyük trafik ışığının ayrı bir pencerede gösterilmesi                    - 7.03.2026 Yapıldı.
- Trafik ışığı kontrolünün belli bir alana(karşı tarafa) sınırlandırılması
- Opencv HSV ile kırmızı ve yeşil ışık tespiti
- Temporal Smoothing eklenecek
- Takip algoritması kullanılacak
- Segmentasyon maskeleri ile arabalara yaklaşılıyor mu tespiti
- Segmentasyon maskeleri ile sol şeritte araba var mı tespiti
- Yol maskesinin ortasına sanal şerit eklenecek  - 8.03.2026 Yapıldı.
- Sanal şeritin stabil hale getirilmesi gerekiyor.
- Yol maskesine eklenen sanal şerit ile Opencv'den alınan şeritler birleştirilecek
- Bu şeritten sapma oranı ekrana yazdırılacak
- Road maskesinin üzerinde person var mı kontrolü yapılacak
Otomatik sürüş
- Şerit takibi
- Kırmızı ışığa yaklaşınca durma
- Arabaya yaklaşılıyorsa durma
- Yolda yaya varsa durma
- Sol tarafa geç komutu verildiğinde uygunluk kontrolü yapılarak sollama yapılması
- Performans optimizasyonları
- Dosya fonksiyonlara parçalanacak
- Dosya modüllere ayrılacak
- Pyqt/Pyside ile arayüz yazılacak

- C++'a geçiş.

# 8.03.2026
## 5. Geometrik dönüşümler
    - cv2.warpPerspective           - 8.03.2026 Yapıldı. Detayına sitedeki dökümantasyonda girilecek.
    - cv2.getRotationMatrix2D       - 8.03.2026 Yapıldı. Detayına sitedeki dökümantasyonda girilecek.       
    - cv2.getAffineTransform 
    - cv2.getPerspectiveTransform
    - cv2.remap

# 9.03.2026

## 6. Görüntü Aritmetiği ve Bitwise İşlemler

    - cv2.add                       - 10.03.2026 Yapıldı. Numpy ile arasındaki overflow vs saturation farkı gösterildi.
    - cv2.subtract                  - 19.03.2026 Yapıldı. Numpy ile arasındaki fark gösterildi.
    - cv2.multiply                  - 22.03.2026 Yapıldı. Numpy ile arasındaki fark gösterildi.
    - cv2.divide                    - 22.03.2026 Yapıldı. Numpy ile arasındaki fark gösterildi.
    - cv2.addWeighted               - 22.03.2026 Yapıldı.
    - cv2.bitwise_and
    - cv2.bitwise_or
    - cv2.bitwise_not
    - cv2.bitwise_xor
    - cv2.absdiff
    - cv2.normalize
    - cv2.convertScaleAbs

---
# 10.03.2026

## 7. Filtreleme ve Bulanıklaştırma

    - cv2.blur
    - cv2.GaussianBlur
    - cv2.medianBlur
    - cv2.bilateralFilter
    - cv2.filter2D
    - cv2.boxFilter
    - cv2.stackBlur
    - cv2.sepFilter2D

# 11.03.2026

## 8. Kenar Tespiti ve Gradyan

    - cv2.Canny
    - cv2.Sobel
    - cv2.Scharr
    - cv2.Laplacian

# 12.03.2026

## 9. Eşikleme (Thresholding)

    - cv2.threshold
    - cv2.adaptiveThreshold

## 10. Morfolojik İşlemler

    - cv2.erode
    - cv2.dilate
    - cv2.morphologyEx
    - cv2.getStructuringElement

# 13.03.2026

    11. Kontur İşlemleri

# 14.03.2026
    11. Kontur İşlemleri
# 15.03.2026
    11. Kontur İşlemleri

---
# 16.03.2026
    12. Histogram İşlemleri
    Proje 1 - Şerit Tespiti
# 17.03.2026
    13. Öznitelik Tespiti (Feature Detection)
# 18.03.2026
    13. Öznitelik Tespiti (Feature Detection)
# 19.03.2026
    14. Nesne Tespiti
# 20.03.2026
    14. Nesne Tespiti
# 21.03.2026
    15. Optik Akış ve Video Analizi
# 22.03.2026
    15. Optik Akış ve Video Analizi
# 23.03.2026
    15. Optik Akış ve Video Analizi
---
# 24.03.2026
    16. Görüntü Dönüşümleri (Transform)
    Proje 2 - Hareketli Nesne takibi
# 25.03.2026
    16. Görüntü Dönüşümleri (Transform)
# 26.03.2026
    16. Görüntü Dönüşümleri (Transform)
# 27.03.2026
    17. Görüntü Segmentasyonu
# 28.03.2026
    17. Görüntü Segmentasyonu
# 29.03.2026
    19. DNN (Derin Sinir Ağı) Modülü
# 30.03.2026
    19. DNN (Derin Sinir Ağı) Modülü
---
# 31.03.2026
    19. DNN (Derin Sinir Ağı) Modülü
# 1.04.2026
    19. DNN (Derin Sinir Ağı) Modülü
# 2.04.2026
    19. DNN (Derin Sinir Ağı) Modülü
    Proje 3 - Trafik tabelası tanıma
# 3.04.2026
    20. GUI ve Etkileşim
# 4.04.2026
    21. Diğer Faydalı Fonksiyonlar
# 5.04.2026
    Proje 4 - Opencv dnn modulü ile nesne tespiti ve uzaklık ölçümü
    Numpy'a giriş
6.04.2026
