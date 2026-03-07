# C++ Pipeline — Gerçek Zamanlı YOLO Segmentasyon

# Notlar
# ⚠️ Bu kod önizlemedir. Çalışılabilirlik garanti edilmez.
# İleride OpenCV CUDA ve custom CUDA kernellerine dönüştürülecektir.

---

## Pipeline Akışı

```
DXGI Capture → Letterbox Preprocess → TensorRT FP16 Inference → NMS → Proto Mask → Overlay
   10.5ms            4.7ms                  14.7ms              ~0ms     5.8ms
                                    Toplam: ~35ms → ~50 FPS
```

**6 sınıf:** `road` · `sidewalk` · `car` · `motorcycle` · `person` · `traffic_light`

---

## Dosya Yapısı

```
C++_Pipeline/
├── src/
│   ├── main.cpp                # Giriş noktası, pencere döngüsü, thread yönetimi
│   └── yolo_seg.cpp            # Tüm pipeline: capture, preprocess, infer, postprocess, mask
├── include/
│   ├── yolo_seg.hpp            # Sınıf tanımı, sabitler, Detection struct
│   └── dxgi_capture.hpp        # DXGI Desktop Duplication API sarmalayıcı
└── README.md                   # Bu dosya
```

---

## Gereksinimler

### Donanım
- NVIDIA GPU (Compute Capability 7.0+)
- Önerilen: RTX 3060 veya üzeri

### Yazılım

| Bağımlılık | Versiyon | İndirme |
|------------|----------|---------|
| Windows | 10 / 11 | — |
| Visual Studio | 2022 (C++ Desktop Development) | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/) |
| CUDA Toolkit | 13.x | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| cuDNN | 9.x (CUDA 13 uyumlu) | [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) |
| TensorRT | 10.x | [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) |
| OpenCV | 4.x (C++ build) | [opencv.org/releases](https://opencv.org/releases/) |

---

## Kurulum

### 1. CUDA Toolkit

[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) indirip kur. Kurulumdan sonra kontrol:

```cmd
nvcc --version
```

### 2. cuDNN

[cuDNN](https://developer.nvidia.com/cudnn) indir, arşivi aç. İçindeki dosyaları CUDA dizinine kopyala:

```
bin\*.dll      →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\bin\
include\*.h    →  ...\include\
lib\x64\*.lib  →  ...\lib\x64\
```

### 3. TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) indir ve bir dizine çıkar. Örnek:

```
C:\TensorRT-10.x.x.x\
```

`bin\` klasörünü sistem PATH'e ekle:

```
C:\TensorRT-10.x.x.x\bin
```

### 4. OpenCV (C++ Build)

[OpenCV Releases](https://opencv.org/releases/) sayfasından Windows için indir ve çıkar:

```
C:\opencv\
```

### 5. Ortam Değişkenleri

Sistem PATH'e eklenecekler:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
C:\TensorRT-10.x.x.x\lib
C:\opencv\build\x64\vc16\bin
```

---

## Derleme

### CMake ile (Önerilen)

> ⚠️ CMake build sistemi henüz hazır değil. Aşağıdaki yapı Bölüm 5'te eklenecek.

Planlanan CMakeLists.txt yapısı:

```cmake
cmake_minimum_required(VERSION 3.18)
project(yolo_seg_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT — Manuel yol belirtme
set(TENSORRT_DIR "C:/TensorRT-10.x.x.x")

add_executable(yolo_seg_trt
    src/main.cpp
    src/yolo_seg.cpp
)

target_include_directories(yolo_seg_trt PRIVATE
    src/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
)

target_link_libraries(yolo_seg_trt PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_DIR}/lib/nvinfer.lib
    ${TENSORRT_DIR}/lib/nvinfer_plugin.lib
    d3d11.lib
    dxgi.lib
)
```

Derleme komutları:

```cmd
mkdir build
cd build
cmake .. -DTENSORRT_DIR="C:/TensorRT-10.x.x.x"
cmake --build . --config Release
```

### Visual Studio ile (Şu anki yöntem)

1. Visual Studio 2022 aç → yeni boş C++ projesi oluştur
2. Proje ayarları → Configuration Properties:

**C/C++ → General → Additional Include Directories:**
```
C:\opencv\build\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include
C:\TensorRT-10.x.x.x\include
```

**Linker → General → Additional Library Directories:**
```
C:\opencv\build\x64\vc16\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64
C:\TensorRT-10.x.x.x\lib
```

**Linker → Input → Additional Dependencies:**
```
opencv_world4100.lib
nvinfer.lib
nvinfer_plugin.lib
cudart.lib
d3d11.lib
dxgi.lib
```

3. Platform: **x64**, Configuration: **Release**
4. Build → Build Solution (Ctrl+Shift+B)

---

## Çalıştırma

### 1. YOLO Engine Dosyası Hazırlama

Önce Ultralytics ile YOLO modelini ONNX'e, sonra TensorRT engine'e çevir:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="engine", half=True, imgsz=640)
```

Bu işlem `best.engine` dosyası üretir.

### 2. Programı Çalıştırma

```cmd
yolo_seg_trt.exe best.engine
```

GTA V 1280x720 pencere modunda çalışıyor olmalı.

---

## Performans Ölçümleri (RTX 5070 Ti)

```
capture=10.5ms  pre=4.7ms  infer=14.7ms  post=0.0ms  mask=5.8ms  dets=12
capture=10.2ms  pre=4.8ms  infer=14.5ms  post=0.0ms  mask=5.9ms  dets=15
capture=10.8ms  pre=4.6ms  infer=14.9ms  post=0.0ms  mask=5.7ms  dets=11
```

| Aşama | Süre | Açıklama |
|-------|------|----------|
| Capture | ~10.5ms | DXGI Desktop Duplication API |
| Preprocess | ~4.7ms | Letterbox + BGR→RGB + Normalize + HWC→NCHW |
| Inference | ~14.7ms | TensorRT FP16 |
| Postprocess | ~0.0ms | Ultralytics NMS (engine içinde) |
| Mask | ~5.8ms | Proto mask hesaplama + overlay |
| **Toplam** | **~35ms** | **~50 FPS (double buffer ile)** |

---

## Pipeline Detayları

### Preprocess (4 adım)
1. **Letterbox:** 1280x720 → 640x640 (aspect ratio korunur, gri padding)
2. **BGR → RGB:** OpenCV BGR olarak okur, model RGB bekler
3. **Normalize:** [0,255] → [0,1] (piksel / 255)
4. **HWC → NCHW:** Satır-sütun-kanal → Kanal-satır-sütun (TensorRT beklentisi)

### Postprocess
- Model çıktısı: `[1, 300, 38]` → her satır: `[x1, y1, x2, y2, conf, cls, mask_coeffs×32]`
- Koordinatlar 640x640 letterbox space'inde → padding çıkar, scale'e böl → orijinal koordinatlar

### Mask
- Proto mask: 32 adet 160x160 şablon
- Her tespit için: `mask = Σ coeff[k] × proto[k]` → sigmoid → threshold(0.5) → overlay

---

## Bilinen Sorunlar

- **Flicker:** Maskelerde frame'den frame'e kararsız tespit. Temporal smoothing ile çözülecek.
- **3. şahıs road tespiti zayıf:** Model 1. şahıs verisiyle eğitildi, fine-tune gerekiyor.
- **Bird eye view:** Denenip geri alındı — mask 5ms → 117ms performans kaybı.

---

## Seri Boyunca Bu Koda Ne Olacak

Bu kod serinin son hali değil, başlangıç noktası. Seri boyunca:

- [ ] Her satıra yorum eklenecek
- [ ] Preprocess Python'da sıfırdan yazılacak, sonra C++'a çevrilecek
- [ ] Tracker entegre edilecek (flicker çözümü)
- [ ] Kontrol mekanizması eklenecek
- [ ] CMake build sistemi kurulacak
- [ ] Performans optimizasyonları yapılacak

---

