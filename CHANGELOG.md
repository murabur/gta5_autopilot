# CHANGELOG

Bu dosya projenin geliştirme sürecindeki teknik ilerlemeleri ve yapılan deneyleri kaydetmek amacıyla tutulmaktadır.
## 2026-03-11
- [NumPy ndarray](./Bolum_1_Temeller/Video_1.2_Numpy/1_numpy_ndarray.ipynb) eklendi
- [NumPy array oluşturma](./Bolum_1_Temeller/Video_1.2_Numpy/2_numpy_array_olusturma.ipynb) eklendi
- [NumPy indexing ve slicing ](./Bolum_1_Temeller/Video_1.2_Numpy/3_numpy_indexing_slicing.ipynb) konusuna başlandı.

## 2026-03-10
- [NumPy dosyası](./Bolum_1_Temeller/Video_1.2_Numpy/0_numpy_dosyasi.ipynb) eklendi.
- Numpy'a başlangıç yapıldı.
- [NumPy array oluşturma yöntemleri](./Bolum_1_Temeller/Video_1.2_Numpy/1_numpy_array_olusturma.ipynb)
-  [OpenCV aritmetik ve bitwise işlemleri ](./Bolum_1_Temeller\Video_1.3_OpenCV\6_aritmetik_bitwise.ipynb) cv2.add fonksiyonu gösterildi. Numpy ile arasındaki overflow vs saturation farkı gösterildi.
- [yolo_detect_v2 ](./YOLO/yolo_detect_v2.py) eklendi. Performans ciddi manada (14 - 18 FPS -> 30 FPS) arttırıldı.
 


## 2026-03-08
- road (id=0) segmentation maskesinin orta noktasını hesaplayan `get_road_centerline(road_mask)` fonksiyonu eklendi.
- Maske üzerinden sanal şerit çizme denemeleri yapıldı.
- Daha stabil hale getirilmesi gerekiyor.

## 2026-03-07
- OpenCV bölümüne aşağıdaki fonksiyonların uygulamaları eklendi:
  - resize
  - flip
  - rotate
  - warpAffine
- Ekrandaki en büyük trafik ışığını ayrı bir pencerede gösteren test eklendi.

## 2026-03-04
- OpenCV bölümüne başlandı.
- Görüntü okuma (image loading) örneği eklendi.