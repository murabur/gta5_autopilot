# GTA V Otopilot

Bana ulaşabileceğiniz bağlantılar

Bu projenin amacı GTA 5'de Autopilot yaparken görüntü işleme ve yapay zeka tekniklerini incelemektir.

### 🌐 Bağlantılar & Sosyal Medya
[![Website](https://img.shields.io/badge/Web_Sitem-muratburc.com-blue?style=for-the-badge&logo=google-chrome&logoColor=white)](https://muratburc.com/)
[![YouTube Ana Kanal](https://img.shields.io/badge/YouTube-Murat_Burç-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@muratburc)
[![YouTube Destek](https://img.shields.io/badge/YouTube-Murat_Burç_Tech-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCcgArUBkEnL3s2F6M7OuWYw)

Çok önemli not: Aşağıdaki plan kesinlik ifade etmemektedir ve "hedef" seviyesindedir. Bu konuların tamamına hakim değilim. Nihai amacım bu konuları öğrenmek ve size doğru şekilde aktarabilmek.

## Planlanan video yol haritası(değişme ihtimali her zaman için mevcut)
### Bölüm 1 - Temeller
- Video 1.1 - Bu serinin amacı
- Video 1.2 - Numpy
- Video 1.3 - OpenCV
- Video 1.4 - PIL vs mss vs Bettercam ekran yakalama karşılaştırması
- Video 1.5 - OpenCV ile tespit denemeleri
- Video 1.6 - Neden derin öğrenme modellerine ihtiyacımız var?
### Bölüm 2 - YOLO İle Bounding Box Custom Object Detection
- Video 2.1 - Kullanılabilecek sinir ağı modelleri, neden YOLO?
    - COCO veri setiyle eğitilmiş YOLO modeliyle oyun içerisinde deneme
- Video 2.2 - YOLO ile uçtan uca Custom Object Detection - Özel Nesne Tespit Rehberi
    - Ultralytics kütüphanesi
    - Pytorch
    - Veri toplama
        - Veri çeşitliliğinin önemi
        - Augmentation - Veri arttırımı
    - Veri etiketleme
        - Veri etiketleme araçları
        - Otomatik/Yarı otomatik etiketleme döngüsü kurmak
    - Dosya/Klasör yapısı
        - Dosya formatı
        - Klasör formatı
        - YAML dosyası
    - Model eğitimi
        - Model metriklerinin yorumlanması   
    - Modelin test edilmesi
    - Otomatik/yarı otomatik veri etiketleme döngüsü ile modelin iyileştirilmesi
- Video 2.3 - YOLO model Optimizasyonu
    - Model boyutu - n - s - x vb.
    - imgsize parametresi
    - Eğitim seti verisinin performans ve doğruluğa etkisi - 500 - 1000 - 2000 - 5000
    - Eğitim parametreleri
        - batch size
        - learning rate
        - optimizasyon algoritması
    - Nvidia GPU'lar için TensorRT optimizasyonu
### Bölüm 3 - YOLO İle Pixel Bazlı Segmentasyon
- Video 3.1 - YOLO ile Pixel Bazlı Segmentasyon
    - Detection vs Segmentation
    - Neden segmentasyon gerekli?
    - COCO veri seti ile eğitilmiş YOLO seg modelini kullanarak oyun içinde tespit gerçekleştirme
- Video 3.2 - Yolo ile Custom Segmentation - Özel Segmentasyon Rehberi
    - Hazır veri setleri
        - Cityscapes
        - BDD100K
        - Playing for Data: Ground Truth from Computer Games(GTA 5 içerisinden hazırlanmış segmentasyon veri seti)
            - Bu veri setindeki maskelerin YOLO .txt formatına dönüştürülmesi
    - Veri toplama
    - Verilerin etiketlenmesi
        - Hazır etiketleme modelleri - Segment Anything vb.
        - Model eğitimi ve otomatik veri etiketleme sisteminin kurulması
    Model eğitimi
        - Model metriklerinin yorumlanması
    - Modelin test edilmesi
    - Otomatik/yarı otomatik veri etiketleme döngüsü ile modelin iyileştirilmesi
- Video 3.3 - Segmentasyon model optimizasyonu
    - Model boyutu - n - s - x vb.
    - imgsize parametresi
    - Eğitim seti verisinin performans ve doğruluğa etkisi - 500 - 1000 - 2000 - 5000
    - Eğitim parametreleri
        - batch size
        - learning rate
        - optimizasyon algoritması
    - Nvidia GPU'lar için TensorRT optimizasyonu
- Video 3.4 - YOLO modellerinin karşılaştırılması
    - YOLO V8 vs YOLO V11 vs YOLO 26
- Video 3.5 - Segmentasyon sonuçlarının görselleştirilmesi
- Video 3.6 - Performans optimizasyonları - Profiling
- Video 3.7 - Segmentasyon ile ilk karar denemeleri
### Bölüm 4 - Kontrol Mekanizması - Not: Burası hala kafamda net değil. Güncellenecek.
- Video 4.1 - Kontrol Algoritmaları
- Video 4.2 - Maske Koordinatları üzerinden ve Open CV yardımıyla şerit tespiti ve şerit takibi
- Video 4.3 - Maske koordinatları üzerinden trafik ışığı tespiti ve HSV renk analizi(kırmızı - yeşil)
- Video 4.4 - Maske koordinatları üzerinden yaya tespiti ve aksiyon alma
- Video 4.5 - Maske koordinatları üzerinden araçlar arası mesafe tespiti
- Video 4.6 - Tüm tespitleri birleştirerek karar alma
- Video 4.7 - İlk sürüş: vgamepad kütüphanesi üzerinden sürüş komutu verme
- Video 4.8 - Karar mekanizmalarının vgameped'e entegre edilmesi
- Video 4.9 - Sollama mekaniği

### Ara bölüm
- PyQT veya Pyside ile arayüz tasarımı
### Bölüm 5 - Karşınızda C++
- Video 5.1 - Neden C++? C++ vs Python performans karşılaştırması
- Video 5.2 - C++'da bilinmesi/aşina olunması gerekenler
    - Visual Studio kurulumu
    - Linker meselesi ve kütüphane kurulumları
- Video 5.3 - DXGI Desktop Duplication API ile ekran yakalama
- Video 5.4 - OpenCV C++
- Video 5.5 - OpenCV CUDA C++
### Bölüm 6 - Zor bölüm - Gelecek hedefleri - Zero Copy Pipeline
- Video 6.1 CUDA bellek yönetimi
- Video 6.2 DX11/12 to CUDA Interop
- Video 6.3 CUDA kernels(preprocessing)
- Video 6.4 TensorRT C++ API
- Nihai hedef: End-to-End GPU pipeline
### Bölüm 8 - Kontrol mekanizması
- Python kontrol kısmının C++'a uygulanması (şimdilik belirsiz)

## Gelecek hedefleri
- YOLO yerine Custom CNN kullanılması
- Vision Transformer


- 4.03.2026 - Open CV bölümüne başlandı. Görüntü okuma dosyası eklendi.