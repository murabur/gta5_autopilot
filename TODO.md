# Model
- Üçüncü şahıs perspektifinden veri toplama ve etiketleme
- Yeni veri setiyle fine tune
- Eğitilen modelin paylaşılması

# Algılama

# Görüntü stabilitesi
- Temporal smoothing                                                                    - Yapıldı

# Yol - Kaldırım
- Yolda - Yolda değil tespiti
- Sanal şerit
    - Sanal şeritin stabil hale getirilmesi
    - Yol maskesine eklenen sanal şerit ile Opencv'den alınan şeritler birleştirilecek
    - Şeritten sapma oranı ekrana yazdırılacak


#Araba - Motorsiklet
- Uzaklaşıyor - Sabit - Yakınlaşıyor tespiti                                            - Yapıldı
- Ana aracın Sol ve sağında araba - motorsiklet var mı tespiti
- Takip algoritması kullanılacak                                                        - Yapıldı

# Trafik ışığı
- Ekrandaki en büyük trafik ışığının ayrı bir pencerede gösterilmesi                    - 7.03.2026 Yapıldı.
- Trafik ışığı kontrolünün belli bir alana(karşı tarafa) sınırlandırılması
- OpenCV HSV ile kırmızı - sarı - yeşil ışık tespiti

# Yaya
- Road maskesinin üzerinde person var mı kontrolü

# Kontrol
- Şerit takibi
- Kırmızı ışığa yaklaşınca durma
- Arabaya yaklaşılıyorsa durma
- Yolda yaya varsa durma
- Sol tarafa geç komutu verildiğinde uygunluk kontrolü yapılarak sollama yapılması

# Kod kalitesi
- Performans optimizasyonları
- Dosya fonksiyonlara parçalanacak
- Dosya modüllere ayrılacak

# Arayüz
- Pyqt/Pyside ile arayüz yazılacak

# C++
- C++'a geçiş.



