# NumPy Python Fonksiyon Referansı (Computer Vision Odaklı)

Toplam kapsam: ~40 kritik fonksiyon + temel kavramlar  
Bu liste NumPy kullanımının yaklaşık %80–90'ını kapsar.

---

# 1. NumPy Temelleri

| Kavram | Açıklama |
|------|------|
| ndarray | NumPy'nin temel veri yapısı |
| shape | Array boyutları |
| ndim | Array kaç boyutlu |
| size | Toplam eleman sayısı |
| dtype | Veri tipi |
| itemsize | Bir elemanın byte boyutu |
| nbytes | Array'in toplam bellek kullanımı |

---

# 2. Array Oluşturma

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.array()` | Python listeden array oluşturur |
| `np.zeros()` | Sıfırlardan oluşan array |
| `np.ones()` | Birlerden oluşan array |
| `np.full()` | Sabit değerli array |
| `np.zeros_like()` | Var olan array boyutunda sıfır array |
| `np.ones_like()` | Var olan array boyutunda bir array |
| `np.arange()` | Aralık üretir |
| `np.linspace()` | Belirli sayıda eşit aralıklı değer |
| `np.eye()` | Identity matrix |
| `np.identity()` | Identity matrix |
| `np.diag()` | Diagonal oluşturur / okur |

---

# 3. Indexing ve Slicing

| Kavram | Açıklama |
|------|------|
| `array[i]` | Tek eleman |
| `array[i,j]` | 2D eleman erişimi |
| `array[:, :]` | Tüm satır ve sütun |
| `array[:, 0]` | Sütun seçme |
| `array[0, :]` | Satır seçme |
| negative indexing | sondan erişim |
| slicing | belirli aralık seçme |

---

# 4. Boolean Masking

| Fonksiyon | Ne Yapıyor |
|------|------|
| `array > value` | Boolean mask üretir |
| `array[mask]` | Mask ile seçim |
| `np.where()` | Koşullu seçim |
| `np.logical_and()` | Mantıksal AND |
| `np.logical_or()` | Mantıksal OR |
| `np.logical_not()` | Mantıksal NOT |

---

# 5. Matematiksel İşlemler

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.add()` | Toplama |
| `np.subtract()` | Çıkarma |
| `np.multiply()` | Çarpma |
| `np.divide()` | Bölme |
| `np.sqrt()` | Kare kök |
| `np.exp()` | Üstel |
| `np.log()` | Logaritma |
| `np.abs()` | Mutlak değer |

---

# 6. Aggregation (Toplama ve İstatistik)

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.sum()` | Toplam |
| `np.mean()` | Ortalama |
| `np.min()` | Minimum |
| `np.max()` | Maximum |
| `np.std()` | Standart sapma |
| `np.var()` | Varyans |
| `np.argmax()` | En büyük eleman index |
| `np.argmin()` | En küçük eleman index |

---

# 7. Axis Mantığı

| Parametre | Açıklama |
|------|------|
| `axis=0` | sütun boyunca |
| `axis=1` | satır boyunca |

---

# 8. Broadcasting

| Örnek | Açıklama |
|------|------|
| `array + scalar` | scalar broadcast |
| `array * scalar` | scalar çarpma |
| farklı boyutlu array işlemleri | broadcast |

---

# 9. Shape Manipülasyonu

| Fonksiyon | Ne Yapıyor |
|------|------|
| `reshape()` | şekil değiştirir |
| `flatten()` | tek boyuta indirir |
| `ravel()` | flatten benzeri |
| `transpose()` | satır sütun değiştirir |
| `swapaxes()` | eksen değiştirir |

---

# 10. Array Birleştirme

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.concatenate()` | array birleştirme |
| `np.vstack()` | dikey birleştirme |
| `np.hstack()` | yatay birleştirme |
| `np.stack()` | yeni eksen ile birleştirme |

---

# 11. Array Bölme

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.split()` | array bölme |
| `np.vsplit()` | dikey bölme |
| `np.hsplit()` | yatay bölme |

---

# 12. Sorting ve Arama

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.sort()` | sıralama |
| `np.argsort()` | sıralama index |
| `np.unique()` | benzersiz elemanlar |

---

# 13. Veri Tipi İşlemleri

| Fonksiyon | Ne Yapıyor |
|------|------|
| `astype()` | veri tipi dönüştürür |
| `np.uint8` | 8 bit integer |
| `np.float32` | 32 bit float |
| `np.float64` | 64 bit float |

---

# 14. Random

| Fonksiyon | Ne Yapıyor |
|------|------|
| `np.random.rand()` | uniform random |
| `np.random.randn()` | normal dağılım |
| `np.random.randint()` | integer random |
| `np.random.uniform()` | uniform dağılım |
| `np.random.normal()` | normal dağılım |

---

# 15. Computer Vision İçin Kritik NumPy Konuları

| Konu | Açıklama |
|------|------|
| image = ndarray | görüntü aslında array |
| channel access | img[:,:,0] |
| ROI slicing | img[y1:y2, x1:x2] |
| boolean mask | segmentation |
| np.where | pixel koordinatları |
| np.mean | centroid hesaplama |
| broadcasting | brightness / contrast |
| dtype | uint8 vs float32 |
| vectorization | performans |

---

# 16. NumPy Performans Mantığı

| Konu | Açıklama |
|------|------|
| vectorization | loop yerine array işlemi |
| contiguous memory | bellek düzeni |
| view vs copy | gereksiz kopyadan kaçınma |

---

# Toplam Çekirdek

NumPy'de gerçek projelerin çoğu şu fonksiyonları kullanır:

- np.array
- np.zeros
- np.ones
- np.arange
- np.linspace
- np.where
- np.mean
- np.sum
- np.min
- np.max
- reshape
- transpose
- concatenate
- astype
- np.random.randint