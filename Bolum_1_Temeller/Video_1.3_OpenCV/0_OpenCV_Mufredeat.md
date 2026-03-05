Takip edilmesi planlanan müfredat budur. Size bunun hepsini yapacağımın garantisini veremiyorum. Çoğunu yapmaya çalışacağım.

# OpenCV Python Fonksiyon Referansı

## 1. Görüntü Okuma / Yazma / Gösterme

| Fonksiyon | Ne Yapıyor |Durum |
|-----------|-----------|-----------| 
| `cv2.imread(path, flag)` | Dosyadan görüntü okur. flag: IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED |[✅ 1_goruntu_okuma.py](./1_goruntu_okuma.py) |
| `cv2.imwrite(path, img)` | Görüntüyü dosyaya yazar (jpg, png, bmp...) |
| `cv2.imshow(name, img)` | Pencerede görüntü gösterir |[✅ 1_goruntu_okuma.py](./1_goruntu_okuma.py) |
| `cv2.waitKey(ms)` | Belirtilen ms kadar tuş bekler. 0 = sonsuza kadar |[✅ 1_goruntu_okuma.py](./1_goruntu_okuma.py) | [✅ 4_video_dosyasindan_goruntu_okuma.py](./4_video_dosyasindan_goruntu_okuma.py)
| `cv2.destroyAllWindows()` | Tüm OpenCV pencerelerini kapatır |[✅ 1_goruntu_okuma.py](./1_goruntu_okuma.py) |
| `cv2.destroyWindow(name)` | Belirli bir pencereyi kapatır |
| `cv2.namedWindow(name, flag)` | Pencere oluşturur. WINDOW_NORMAL = boyutlandırılabilir |
| `cv2.resizeWindow(name, w, h)` | Pencere boyutunu ayarlar |
| `cv2.getWindowProperty(name, prop)` | Pencere özelliğini sorgular (açık mı kapalı mı vs.) |
| `cv2.moveWindow(name, x, y)` | Pencereyi ekranda taşır |

## 2. Video Okuma / Yazma / Kamera

| Fonksiyon | Ne Yapıyor | Durum |
|-----------|-----------|-----------| 
| `cv2.VideoCapture(source)` | Video dosyası veya kamera açar (0 = varsayılan kamera) | [✅ 3_kameradan_goruntu_okuma.py](./3_kameradan_goruntu_okuma.py)
| `cap.read()` | Bir frame okur → (bool, frame) döndürür | [✅ 3_kameradan_goruntu_okuma.py](./3_kameradan_goruntu_okuma.py)
| `cap.isOpened()` | Video/kamera açık mı kontrol eder |
| `cap.release()` | Video/kamera kaynağını serbest bırakır | [✅ 3_kameradan_goruntu_okuma.py](./3_kameradan_goruntu_okuma.py)
| `cap.get(propId)` | Video özelliğini okur (FPS, genişlik, yükseklik...) |
| `cap.set(propId, value)` | Video özelliğini ayarlar |
| `cv2.VideoWriter(path, fourcc, fps, size)` | Video dosyası yazar |
| `cv2.VideoWriter_fourcc(*codec)` | Video codec belirler ('XVID', 'mp4v', 'MJPG'...) |
| `writer.write(frame)` | Frame'i videoya yazar |
| `writer.release()` | Video yazıcıyı kapatır |

## 3. Çizim Fonksiyonları

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.line(img, pt1, pt2, color, thickness)` | Çizgi çizer |
| `cv2.rectangle(img, pt1, pt2, color, thickness)` | Dikdörtgen çizer. thickness=-1 → dolgulu |
| `cv2.circle(img, center, radius, color, thickness)` | Daire çizer |
| `cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)` | Elips çizer |
| `cv2.polylines(img, pts, isClosed, color, thickness)` | Çokgen çizer |
| `cv2.fillPoly(img, pts, color)` | Dolgulu çokgen çizer |
| `cv2.putText(img, text, org, font, scale, color, thickness, lineType)` | Metin yazar |
| `cv2.getTextSize(text, font, scale, thickness)` | Metin boyutunu hesaplar → (w,h), baseline |
| `cv2.arrowedLine(img, pt1, pt2, color, thickness)` | Oklu çizgi çizer |
| `cv2.drawMarker(img, pos, color, markerType, markerSize, thickness)` | İşaretçi çizer |
| `cv2.drawContours(img, contours, idx, color, thickness)` | Konturları çizer |

## 4. Renk Dönüşümleri

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.cvtColor(img, code)` | Renk uzayı dönüşümü (BGR↔RGB, BGR↔HSV, BGR↔GRAY...) |
| `cv2.inRange(img, lower, upper)` | Belirli renk aralığındaki pikselleri maskeler (HSV filtre) |
| `cv2.applyColorMap(img, colormap)` | Renk haritası uygular (COLORMAP_JET, COLORMAP_HOT...) |
| `cv2.merge(channels)` | Kanalları birleştirir [B, G, R] → BGR |
| `cv2.split(img)` | Kanalları ayırır BGR → [B, G, R] |
| `cv2.mixChannels(src, dst, fromTo)` | Kanalları karıştırır |

### Yaygın Dönüşüm Kodları
- `cv2.COLOR_BGR2GRAY` — Renkli → Gri
- `cv2.COLOR_BGR2RGB` — BGR → RGB (matplotlib için)
- `cv2.COLOR_BGR2HSV` — BGR → HSV (renk filtresi için)
- `cv2.COLOR_HSV2BGR` — HSV → BGR
- `cv2.COLOR_BGR2LAB` — BGR → LAB
- `cv2.COLOR_BGR2HLS` — BGR → HLS
- `cv2.COLOR_GRAY2BGR` — Gri → BGR (3 kanallı)

## 5. Geometrik Dönüşümler

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.resize(img, dsize, fx, fy, interpolation)` | Boyut değiştirme |
| `cv2.flip(img, flipCode)` | Aynalama. 0=dikey, 1=yatay, -1=ikisi |
| `cv2.rotate(img, rotateCode)` | 90/180/270 derece döndürme |
| `cv2.warpAffine(img, M, dsize)` | Afin dönüşüm (döndürme, kaydırma, ölçekleme) |
| `cv2.warpPerspective(img, M, dsize)` | Perspektif dönüşüm (kuş bakışı görüntü) |
| `cv2.getRotationMatrix2D(center, angle, scale)` | Döndürme matrisi oluşturur |
| `cv2.getAffineTransform(src, dst)` | Afin dönüşüm matrisi (3 nokta) |
| `cv2.getPerspectiveTransform(src, dst)` | Perspektif matrisi (4 nokta) |
| `cv2.remap(img, map1, map2, interpolation)` | Piksel haritası ile dönüşüm |

### İnterpolasyon Yöntemleri
- `cv2.INTER_NEAREST` — En yakın komşu (hızlı, kalitesiz)
- `cv2.INTER_LINEAR` — Bilineer (varsayılan, dengeli)
- `cv2.INTER_CUBIC` — Bikübik (yavaş, kaliteli)
- `cv2.INTER_AREA` — Alan bazlı (küçültme için ideal)
- `cv2.INTER_LANCZOS4` — Lanczos (en kaliteli, en yavaş)

## 6. Görüntü Aritmetiği ve Bitwise İşlemler

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.add(img1, img2)` | Piksel bazlı toplama (saturate) |
| `cv2.subtract(img1, img2)` | Piksel bazlı çıkarma |
| `cv2.multiply(img1, img2)` | Piksel bazlı çarpma |
| `cv2.divide(img1, img2)` | Piksel bazlı bölme |
| `cv2.addWeighted(img1, alpha, img2, beta, gamma)` | Alpha blending: img1*alpha + img2*beta + gamma |
| `cv2.bitwise_and(img1, img2, mask)` | Piksel bazlı AND — maske uygulama |
| `cv2.bitwise_or(img1, img2, mask)` | Piksel bazlı OR |
| `cv2.bitwise_not(img)` | Piksel bazlı NOT — renk tersine çevirme |
| `cv2.bitwise_xor(img1, img2)` | Piksel bazlı XOR |
| `cv2.absdiff(img1, img2)` | Mutlak fark (hareket tespiti için) |
| `cv2.normalize(src, dst, alpha, beta, norm_type)` | Değer aralığını normalize etme |
| `cv2.convertScaleAbs(src, alpha, beta)` | Ölçekle + mutlak değer al → uint8 |

## 7. Filtreleme ve Bulanıklaştırma

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.blur(img, ksize)` | Ortalama bulanıklaştırma |
| `cv2.GaussianBlur(img, ksize, sigmaX)` | Gauss bulanıklaştırma (en yaygın) |
| `cv2.medianBlur(img, ksize)` | Medyan bulanıklaştırma (tuz-biber gürültüsü için) |
| `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)` | İki taraflı filtre (kenarları korur) |
| `cv2.filter2D(img, ddepth, kernel)` | Özel çekirdekle konvolüsyon |
| `cv2.boxFilter(img, ddepth, ksize)` | Kutu filtresi |
| `cv2.stackBlur(img, ksize)` | Stack blur (Gauss'a yakın, daha hızlı) |
| `cv2.sepFilter2D(img, ddepth, kernelX, kernelY)` | Ayrışmış 2D filtre |

## 8. Kenar Tespiti ve Gradyan

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.Canny(img, threshold1, threshold2)` | Canny kenar tespiti |
| `cv2.Sobel(img, ddepth, dx, dy, ksize)` | Sobel gradyan (x veya y yönü) |
| `cv2.Scharr(img, ddepth, dx, dy)` | Scharr gradyan (Sobel'den daha hassas) |
| `cv2.Laplacian(img, ddepth)` | Laplacian (ikinci türev, tüm yönler) |

## 9. Eşikleme (Thresholding)

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.threshold(img, thresh, maxval, type)` | Sabit eşikleme |
| `cv2.adaptiveThreshold(img, maxval, method, type, blockSize, C)` | Adaptif eşikleme |

### Eşikleme Tipleri
- `cv2.THRESH_BINARY` — Eşiğin üstü maxval, altı 0
- `cv2.THRESH_BINARY_INV` — Tersi
- `cv2.THRESH_TRUNC` — Eşiğin üstünü kes
- `cv2.THRESH_TOZERO` — Eşiğin altını sıfırla
- `cv2.THRESH_OTSU` — Otomatik optimal eşik (Otsu)

## 10. Morfolojik İşlemler

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.erode(img, kernel, iterations)` | Aşındırma — beyaz alanları küçültür |
| `cv2.dilate(img, kernel, iterations)` | Genişletme — beyaz alanları büyütür |
| `cv2.morphologyEx(img, op, kernel)` | Morfolojik işlem uygular |
| `cv2.getStructuringElement(shape, ksize)` | Çekirdek oluşturur (MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS) |

### Morfolojik İşlem Tipleri
- `cv2.MORPH_OPEN` — Açma (erode + dilate) — küçük gürültüyü siler
- `cv2.MORPH_CLOSE` — Kapama (dilate + erode) — küçük delikleri kapatır
- `cv2.MORPH_GRADIENT` — Gradyan (dilate - erode) — kenar bulma
- `cv2.MORPH_TOPHAT` — Üst şapka (orijinal - açma) — parlak detaylar
- `cv2.MORPH_BLACKHAT` — Siyah şapka (kapama - orijinal) — karanlık detaylar

## 11. Kontur İşlemleri

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.findContours(img, mode, method)` | Konturları bulur |
| `cv2.drawContours(img, contours, idx, color, thickness)` | Konturları çizer |
| `cv2.contourArea(contour)` | Kontur alanını hesaplar |
| `cv2.arcLength(contour, closed)` | Kontur çevresini hesaplar |
| `cv2.boundingRect(contour)` | Saran dikdörtgen → (x, y, w, h) |
| `cv2.minAreaRect(contour)` | Minimum alan dikdörtgen (döndürülmüş) |
| `cv2.minEnclosingCircle(contour)` | Minimum saran daire |
| `cv2.fitEllipse(contour)` | Elips uydurma |
| `cv2.fitLine(contour, distType, param, reps, aeps)` | Çizgi uydurma |
| `cv2.convexHull(contour)` | Dışbükey zarf |
| `cv2.isContourConvex(contour)` | Dışbükey mi kontrol eder |
| `cv2.approxPolyDP(contour, epsilon, closed)` | Konturu poligona yaklaştırır |
| `cv2.moments(contour)` | Kontur momentlerini hesaplar (ağırlık merkezi vs.) |
| `cv2.matchShapes(contour1, contour2, method, parameter)` | İki konturu karşılaştırır |
| `cv2.pointPolygonTest(contour, pt, measureDist)` | Nokta kontur içinde mi |
| `cv2.connectedComponents(img)` | Bağlı bileşenleri etiketler |
| `cv2.connectedComponentsWithStats(img)` | Bağlı bileşenler + istatistikler |

## 12. Histogram İşlemleri

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.calcHist(images, channels, mask, histSize, ranges)` | Histogram hesaplar |
| `cv2.equalizeHist(img)` | Histogram eşitleme (kontrast artırma) |
| `cv2.calcBackProject(images, channels, hist, ranges, scale)` | Geri projeksiyon |
| `cv2.compareHist(hist1, hist2, method)` | İki histogramı karşılaştırır |
| `cv2.createCLAHE(clipLimit, tileGridSize)` | Adaptif histogram eşitleme (CLAHE) |

## 13. Öznitelik Tespiti (Feature Detection)

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.goodFeaturesToTrack(img, maxCorners, quality, minDistance)` | Shi-Tomasi köşe tespiti |
| `cv2.cornerHarris(img, blockSize, ksize, k)` | Harris köşe tespiti |
| `cv2.cornerSubPix(img, corners, winSize, zeroZone, criteria)` | Alt-piksel köşe hassasiyeti |
| `cv2.ORB_create()` | ORB öznitelik dedektörü (ücretsiz SIFT alternatifi) |
| `cv2.SIFT_create()` | SIFT öznitelik dedektörü |
| `cv2.AKAZE_create()` | AKAZE öznitelik dedektörü |
| `cv2.BFMatcher(normType, crossCheck)` | Brute-Force eşleştirici |
| `cv2.FlannBasedMatcher(indexParams, searchParams)` | FLANN tabanlı hızlı eşleştirici |
| `cv2.drawKeypoints(img, keypoints, outImg)` | Anahtar noktaları çizer |
| `cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg)` | Eşleşmeleri çizer |

## 14. Nesne Tespiti

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.CascadeClassifier(path)` | Haar Cascade sınıflandırıcı yükler |
| `cascade.detectMultiScale(img, scaleFactor, minNeighbors)` | Nesne tespit eder (yüz, göz vs.) |
| `cv2.HOGDescriptor()` | HOG tanımlayıcı (yaya tespiti) |
| `cv2.matchTemplate(img, templ, method)` | Şablon eşleştirme |
| `cv2.minMaxLoc(result)` | Min/max değer ve konum bulma |
| `cv2.groupRectangles(rectList, groupThreshold)` | Üst üste binen dikdörtgenleri birleştirir |

## 15. Optik Akış ve Video Analizi

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.calcOpticalFlowFarneback(prev, next, flow, ...)` | Yoğun optik akış (tüm pikseller) |
| `cv2.calcOpticalFlowPyrLK(prev, next, prevPts, ...)` | Seyrek optik akış (belirli noktalar) |
| `cv2.createBackgroundSubtractorMOG2()` | Arka plan çıkarma (MOG2) |
| `cv2.createBackgroundSubtractorKNN()` | Arka plan çıkarma (KNN) |
| `cv2.meanShift(probImage, window, criteria)` | MeanShift takip |
| `cv2.CamShift(probImage, window, criteria)` | CamShift takip (dönen nesneler) |

## 16. Görüntü Dönüşümleri (Transform)

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.HoughLines(img, rho, theta, threshold)` | Hough çizgi tespiti |
| `cv2.HoughLinesP(img, rho, theta, threshold, minLength, maxGap)` | Olasılıksal Hough çizgi tespiti |
| `cv2.HoughCircles(img, method, dp, minDist, ...)` | Hough daire tespiti |
| `cv2.dft(img, flags)` | Ayrık Fourier dönüşümü |
| `cv2.idft(img, flags)` | Ters Fourier dönüşümü |
| `cv2.magnitude(x, y)` | Büyüklük hesaplama |
| `cv2.phase(x, y)` | Faz hesaplama |
| `cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma)` | Gabor filtre çekirdeği |

## 17. Görüntü Segmentasyonu

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.watershed(img, markers)` | Watershed segmentasyon |
| `cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)` | GrabCut ön plan/arka plan ayırma |
| `cv2.kmeans(data, K, criteria, attempts, flags)` | K-means kümeleme |
| `cv2.distanceTransform(img, distanceType, maskSize)` | Mesafe dönüşümü |
| `cv2.floodFill(img, mask, seedPoint, newVal)` | Taşkın dolgu (paint bucket) |

## 18. Kamera Kalibrasyonu ve 3D

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.findChessboardCorners(img, patternSize)` | Satranç tahtası köşelerini bulur |
| `cv2.calibrateCamera(objPoints, imgPoints, imageSize, ...)` | Kamera kalibrasyonu |
| `cv2.undistort(img, cameraMatrix, distCoeffs)` | Lens bozulmasını düzeltir |
| `cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)` | Poz tahmini |
| `cv2.projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs)` | 3D→2D projeksiyon |
| `cv2.stereoCalibrate(...)` | Stereo kamera kalibrasyonu |
| `cv2.StereoBM_create(numDisparities, blockSize)` | Stereo eşleştirme (derinlik haritası) |

## 19. DNN (Derin Sinir Ağı) Modülü

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.dnn.readNet(model, config)` | Model yükler (ONNX, Caffe, TensorFlow, Darknet) |
| `cv2.dnn.readNetFromONNX(path)` | ONNX model yükler |
| `cv2.dnn.readNetFromDarknet(cfg, weights)` | Darknet/YOLO model yükler |
| `cv2.dnn.blobFromImage(img, scalefactor, size, mean, swapRB, crop)` | Görüntüyü blob'a çevirir (model girişi) |
| `net.setInput(blob)` | Model girişini ayarlar |
| `net.forward(outputNames)` | İleri yayılım — çıktıyı hesaplar |
| `net.setPreferableBackend(backendId)` | Backend seçimi (CUDA, OpenVINO...) |
| `net.setPreferableTarget(targetId)` | Hedef cihaz (CPU, GPU, FPGA...) |
| `cv2.dnn.NMSBoxes(bboxes, scores, confThreshold, nmsThreshold)` | Non-Maximum Suppression |

## 20. GUI ve Etkileşim

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.setMouseCallback(windowName, callback)` | Fare olaylarını yakalar |
| `cv2.createTrackbar(name, windowName, value, count, onChange)` | Kaydırıcı oluşturur |
| `cv2.getTrackbarPos(name, windowName)` | Kaydırıcı değerini okur |
| `cv2.setTrackbarPos(name, windowName, pos)` | Kaydırıcı değerini ayarlar |
| `cv2.selectROI(windowName, img)` | Fare ile ROI (ilgi bölgesi) seçimi |
| `cv2.selectROIs(windowName, img)` | Birden fazla ROI seçimi |

## 21. Diğer Faydalı Fonksiyonlar

| Fonksiyon | Ne Yapıyor |
|-----------|-----------|
| `cv2.copyMakeBorder(img, top, bottom, left, right, borderType)` | Kenarlık ekleme (padding) |
| `cv2.hconcat([img1, img2])` | Yatay birleştirme |
| `cv2.vconcat([img1, img2])` | Dikey birleştirme |
| `cv2.getTickCount()` | Zamanlayıcı başlat |
| `cv2.getTickFrequency()` | Zamanlayıcı frekansı |
| `cv2.useOptimized()` | Optimizasyon aktif mi kontrol |
| `cv2.setUseOptimized(True)` | Optimizasyonu aç/kapa |
| `cv2.getBuildInformation()` | OpenCV build bilgisi (CUDA, TBB vs.) |
| `cv2.cuda.getCudaEnabledDeviceCount()` | CUDA GPU sayısı |
| `cv2.samples.findFile(filename)` | OpenCV örnek dosyalarını bulur |

---

**Toplam: ~180+ fonksiyon, 21 kategori**
