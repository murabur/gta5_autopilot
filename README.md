# GTA V Otopilot

## Miniconda - VS Code kurulumu - YOLO Custom Object Detection rehberi
*  **https://muratburc.com/custom-object-detection/**

## 🛠️ Geliştirme ortamı
* **Conda Env:** `gta5_autopilot_v1`
* **Python:** 3.10

### Setup
```bash
conda create -n gta5_autopilot_v1 python=3.10 -y
conda activate gta5_autopilot_v1
pip install numpy opencv-python pillow mss bettercam
```
## 💻 Donanım
* **CPU:** AMD Ryzen 5600X
* **GPU:** Nvidia RTX 5070 Ti
* **RAM:** 32 GB
* **Çözünürlük:** 1280x720 (Yakalama alanı)

| Metod | Avg FPS | 
| :--- | :--- |
| **Pillow** | 12 FPS |
| **MSS** | 60 FPS |
| **Bettercam** | 170 FPS | 

### 🚀 Ekran Yakalama Performans Karşılaştırması
Farklı kütüphanelerin FPS değerleri arasındaki fark aşağıda gösterilmiştir:

![FPS Kıyaslama Tablosu](Bolum_1_Ekran_Yakalama/PIL%20vs%20mss%20vs%20Bettercam.png)


- 4.03.2026 - Open CV bölümüne başlandı. Görüntü okuma dosyası eklendi.