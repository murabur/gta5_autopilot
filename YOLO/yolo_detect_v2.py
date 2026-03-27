import bettercam
import cv2
import time
import numpy as np
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# 1. Model ve yapılandırma
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = r"YOLO\best.engine"
model = YOLO(MODEL_PATH, task="segment")    #.engine dosyası verildiğinde task="segment" zorunlu

#camera işlemleri
camera = bettercam.create(output_color="BGR")   #opencv BGR istediği için output_color="BGR"
capture_area = (0, 40, 1280, 760)               #pencerenin üst kısmı alınmasın diye 40 - 760 -> 1280 x 720 çözünürlük

CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'traffic_light'}    #Modelin eğitildiği sınıf ID'lerina karşılık gelen isimler
CLASS_COLORS = {
    0: (255, 0, 255),
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 165, 255),
    4: (0, 255, 0),
    5: (0, 0, 255)
}

prev_center_pts = None

# Mesafe ve durum takibi için global sözlükler
vehicle_distances = {}
vehicle_states = {}
vehicle_areas = {}

target_h = 720      #yükseklik
target_w = 1280     #hgenişlik
small_h = 180       #maske işlemlerinde performans için kullanılacak matrisin yüksekliği
small_w = 320       #maske işlemlerinde performans için kullanılacak matrisin genişliği
scale_y = small_h / target_h        #180/720 = 0.25
scale_x = small_w / target_w        #320/1280 = 0.25
global_overlay = np.zeros((small_h, small_w, 3), dtype=np.uint8) #segmentasyon maskelerini, ana görüntü üzerine yarı şeffaf bir şekilde ve en yüksek performansta çizmek için kullanılan matris

# ══════════════════════════════════════════════════════════════════════════════
# 2. Fonksiyonlar
# ══════════════════════════════════════════════════════════════════════════════

#road maskesinin ortasını bulma - sanal şerit
def get_road_centerline(road_mask):
    height, width = road_mask.shape
    center_points = []
    prev_center_x = width // 2

    for y in range(int(height * 0.3), height, 10):
        row = road_mask[y, :]
        white_pixels = np.where(row > 0.5)[0]

        if len(white_pixels) > 0:
            center_x = int(np.mean(white_pixels))
            prev_center_x = center_x
        else:
            center_x = prev_center_x

        center_points.append((center_x, y))

    return center_points


#ekran yakalama fonksiyonu
def screen_capture(cam_obj, area):
    frame = cam_obj.grab(region=area)
    if frame is None: #bettercam bazen None döndürebiliyor. Bunun için eklenen kontrol.
        return None  
    return frame        #None dönmediyse frame'i döndürüyor.


#tahmin yapan fonksiyon
def get_predictions(source):
    results = model.track(  source=source,      #predict yapılan görüntü
                            conf=0.3,           #güven skoru değeri
                            verbose=False,      #terminale yazdırmayı kapatma     
                            half=True,          #FP16 half precision - performans arttırır
                            persist=True,       #takip için hafıza ayarı, false olursa takip kaybında araç yeni bir ID alır
                            stream=True         #model sadece o anki kareyi işler, bir sonraki kareye geçtiğinde eskisini bellekten atar
                            )
    return next(results)


ego_track_id = None

#binilen aracı bulan fonksiyon
def find_ego_car(boxes, classes, track_ids):
    global ego_track_id  #global scope'daki ana değişkene bağlar - yazma yetkisi verir.
    
    if track_ids is None: #track_ids nesnelere atanan kimlik numaraları, eğer None ise return None ile none döndür.
        return None
    
    if ego_track_id is not None:        #Eğer sistem daha önceki karelerde bindiğimiz aracı zaten tespit edip bir ID atadıysa
        if ego_track_id in track_ids:   #ve bu hafızadaki ID, şu anki güncel kamera görüntüsünde (track_ids listesinde) hala mevcutsa
            return ego_track_id         #aşağıdaki işlemlere girmeden direkt bu ID'yi döndür.
    
    best_score = 0  #en iyi aday aramaya en alttan başlanır
    best_id = None  #geçici olarak kimlik tutucu
    
    screen_center_x = 640       #1280 piksel genişliğindeki bir ekranın tam yatay merkezi 1280/640
    screen_bottom = 720         #ekranın en alt piksel satırı.
    
    for i, box in enumerate(boxes):     #tespit edilen kutuları tek tek döngüye sokar.
        x1, y1, x2, y2 = box            #kutunun köşe koordinatlarını değişkenlere atar
        
        #sınıf filtresi
        if classes[i] != 2:     #eğitim setinde car id : 2
            continue            #eğer araba değilse bir sonrakine geçer
  
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if center_y < screen_bottom * 0.5: #
            continue
        
        area = (x2 - x1) * (y2 - y1)
        
        dist_x = abs(center_x - screen_center_x)
        dist_y = abs(center_y - screen_bottom)
        closeness = 1 / (dist_x + dist_y + 1)
        
        score = closeness * closeness * area
        if score > best_score:
            best_score = score
            best_id = track_ids[i]
    
    ego_track_id = best_id
    return ego_track_id

#trafik ışığı tespiti
def detect_traffic_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red = cv2.countNonZero(red1 + red2)
    
    green = cv2.countNonZero(cv2.inRange(hsv, (40, 100, 100), (80, 255, 255)))
    
    yellow = cv2.countNonZero(cv2.inRange(hsv, (15, 100, 100), (35, 255, 255)))
    
    counts = {"KIRMIZI": red, "YESIL": green, "SARI": yellow}
    best = max(counts, key=counts.get)
    
    if counts[best] < 10:
        return "BELIRSIZ", (200, 200, 200)
    
    color_map = {
        "KIRMIZI": (0, 0, 255),
        "YESIL": (0, 255, 0),
        "SARI": (0, 255, 255)
    }
    return best, color_map[best]


#maske işlemleri
def process_lane_data(results, target_h, target_w, annotated_frame, overlay):
    global prev_center_pts
    road_mask_full = None

    if results.masks is not None:
        raw_masks_xy = results.masks.xy
        classes_for_masks = results.boxes.cls.cpu().numpy().astype(int)

        best_road_mask = None
        best_road_area = 0

        for i, mask_pts in enumerate(raw_masks_xy):
            if len(mask_pts) == 0: continue
            class_id = classes_for_masks[i]

            if class_id in [0, 1, 2, 5]:
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                pts = np.array(mask_pts, np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)

                if class_id == 0:
                    area = cv2.contourArea(pts)
                    if area > best_road_area:
                        best_road_area = area
                        best_road_mask = pts

        if best_road_mask is not None:
            temp_road = np.zeros((small_h, small_w), dtype=np.uint8)
            cv2.fillPoly(temp_road, [best_road_mask], 255)
            road_mask_full = cv2.resize(temp_road, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            current_center_pts = get_road_centerline(temp_road)
            current_center_pts = [(int(x / scale_x), int(y / scale_y)) for x, y in current_center_pts]

            if prev_center_pts is not None and len(current_center_pts) > 1:
                min_len = min(len(current_center_pts), len(prev_center_pts))
                smoothed_pts = []
                for k in range(min_len):
                    new_x = int(prev_center_pts[k][0] * 0.7 + current_center_pts[k][0] * 0.3)
                    new_y = current_center_pts[k][1]
                    smoothed_pts.append((new_x, new_y))
                if len(current_center_pts) > min_len:
                    smoothed_pts.extend(current_center_pts[min_len:])
                current_center_pts = smoothed_pts

            prev_center_pts = current_center_pts

            if len(current_center_pts) > 1:
                for j in range(len(current_center_pts) - 1):
                    cv2.line(annotated_frame, current_center_pts[j], current_center_pts[j+1], (0, 255, 255), 2)

        overlay_resized = cv2.resize(overlay, (target_w, target_h))
        cv2.addWeighted(src1=overlay_resized, alpha=0.4, src2=annotated_frame, beta=0.6, gamma=0, dst=annotated_frame)

    return annotated_frame, road_mask_full


#bounding box işlemleri
def draw_detections(results, current_frame, original_frame, road_mask_full):
    global vehicle_distances, vehicle_states
    best_light_roi = None
    ego_id = None
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        track_ids = results.boxes.id
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
            
        ego_id = find_ego_car(boxes, classes, track_ids)
        
        # Bellek temizliği (Ekrandan çıkan araçların verilerini siler)
        current_frame_ids = set(track_ids) if track_ids is not None else set()
        for k in list(vehicle_distances.keys()):
            if k not in current_frame_ids:
                del vehicle_distances[k]
                if k in vehicle_states:
                    del vehicle_states[k]

        # Ego aracın merkez koordinatlarını bul
        ego_cx, ego_cy = None, None
        if ego_id is not None:
            for i, tid in enumerate(track_ids):
                if tid == ego_id:
                    x1, y1, x2, y2 = boxes[i]
                    ego_cx = (x1 + x2) / 2
                    ego_cy = (y1 + y2) / 2
                    break

        max_area = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = classes[i]
            conf = confidences[i]
            name = CLASS_NAMES.get(class_id, "Bilinmeyen")
            tid = track_ids[i] if track_ids is not None else "?"

            # Trafik ışığı takibi
            # Trafik ışığı takibi ve renk tespiti
            if name == "traffic_light":
                current_area = (x2 - x1) * (y2 - y1)
                if current_area > max_area:
                    max_area = current_area
                    best_light_roi = original_frame[y1:y2, x1:x2].copy()
                
                # Her trafik ışığının rengini tespit et
                light_roi = original_frame[y1:y2, x1:x2]
                if light_roi.size > 0:
                    light_color_name, light_color_bgr = detect_traffic_light_color(light_roi)
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), light_color_bgr, 2)
                    cv2.putText(current_frame, f"#{tid} {light_color_name}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color_bgr, 2)

            # Mesafe ve durum hesaplama (Sadece diğer araçlar ve motorlar için)
            relation_text = ""
            relation_color = (255, 255, 255)
            # Başlangıçta global olarak tanımlaman gereken yeni sözlük
            # vehicle_areas = {} 

            if tid != "?" and tid != ego_id and class_id in [2, 3]:
                # Aracın o anki piksel alanı
                current_area = (x2 - x1) * (y2 - y1)
                
                if tid in vehicle_areas:
                    # Önceki 5 karenin alan ortalamasını tutan bir liste varsayımıyla (basitleştirilmiş versiyon)
                    prev_area = vehicle_areas[tid]
                    
                    # Alan değişimi (Delta)
                    area_diff = current_area - prev_area
                    
                    # Titremeyi filtrelemek için alanın en az %2'si kadar bir değişim bekliyoruz
                    threshold = prev_area * 0.02 
                    
                    if area_diff > threshold:
                        vehicle_states[tid] = "Yaklasiyor"
                        relation_color = (0, 0, 255)
                    elif area_diff < -threshold:
                        vehicle_states[tid] = "Uzaklasiyor"
                        relation_color = (0, 255, 0)
                    else:
                        vehicle_states[tid] = "Sabit"
                        relation_color = (0, 255, 255)
                        
                    # Yumuşatılmış güncelleme (Low-pass filter mantığı)
                    vehicle_areas[tid] = (prev_area * 0.7) + (current_area * 0.3)
                else:
                    vehicle_states[tid] = "Takipte"
                    vehicle_areas[tid] = current_area
                    
                relation_text = f"{vehicle_states.get(tid, '')}"
                
                # Duruma göre renk belirleme
                if relation_text == "Yaklasiyor":
                    relation_color = (0, 0, 255) # Kırmızı
                elif relation_text == "Uzaklasiyor":
                    relation_color = (0, 255, 0) # Yeşil
                else:
                    relation_color = (0, 255, 255) # Sarı

            # Çizim işlemleri
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            if name not in ["road", "sidewalk", "traffic_light"]:
                if track_ids is not None and track_ids[i] == ego_id:
                    ego_bottom_x = (x1 + x2) // 2
                    ego_bottom_y = min(y2, target_h - 1)
                    
                    if road_mask_full is not None and road_mask_full[ego_bottom_y, ego_bottom_x] > 0:
                        ego_status = "YOLDA"
                        status_color = (0, 255, 0)
                    else:
                        ego_status = "YOLDA DEGIL"
                        status_color = (0, 0, 255)
                    
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(current_frame, f"EGO #{tid} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(current_frame, ego_status, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                else:
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(current_frame, f"#{tid} {name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Yaklaşıyor / Uzaklaşıyor bilgisini ekrana basma
                    if relation_text:
                        cv2.putText(current_frame, relation_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, relation_color, 2)

    return current_frame, best_light_roi, ego_id



# ══════════════════════════════════════════════════════════════════════════════
# 3. Ana döngü
# ══════════════════════════════════════════════════════════════════════════════
while True:
    t0 = time.perf_counter() #FPS sayaç başlangıç noktası
    
    #ekran yakalama ve frame'e aktarma işlemi
    frame = screen_capture(camera, capture_area)
    if frame is None: continue

    #tahmin işlemi
    time_predict_0 = time.perf_counter()
    results = get_predictions(frame)
    time_predict_1 = time.perf_counter()


    annotated_frame = frame         #frame değişkenine ikinci bir isim atanıyor. Referans atamasıdır. Bellekte yeni bir klopya oluşturmaz. Shallow Reference(Sığ referans) örneğidir.
    original_frame = frame.copy()   #orijinal frame'in kopyası alınıyor. Bellekte yeni bir alan tahsis edilir. Trafik ışıklarının temiz tespiti için kullanılacak.
    global_overlay.fill(0)          #matrisin içindeki tüm hücrelerin değerlerini 0 yapar. Maske matrisini temizler.

    mask_time_0 = time.perf_counter()
    #process_lane_data fonksiyonu maske verilerini döndürür.
    annotated_frame, road_mask_full = process_lane_data(results = results,      #results: YOLO'dan gelen veri nesnesi
                                                        target_h = target_h,    #target_h = 720      #yükseklik
                                                        target_w = target_w,    #target_w = 1280     #hgenişlik
                                                        annotated_frame = annotated_frame,  #annotated_frame fonksiyona veriliyor, fonksiyonda işleniyor, geri tekrar annotated_frame olarak return ediliyor.
                                                        overlay = global_overlay     #global_overlay parametresi ile boş matris fonksiyona veriliyor.
                                                        )
    #annotated_frame işlenmiş görüntüdür. Orjinal frame görüntüsüne referanstır. Değişiklikler frame üzerinde de uygulanır.
    # road_mask_full yolun nerede olduğuna dair saf bilgidir. annotated_frame insan için görsel sonuçken, road_mask_full hesaplama için veridir.
    mask_time_1 = time.perf_counter()

    box_time_0 = time.perf_counter()
    final_display, best_light_roi, ego_id = draw_detections(results = results,                  #YOLO results nesnesi
                                                            current_frame = annotated_frame,    #kutuların çizilmesi için annotated_frame current_frame olarak veriliyor
                                                            original_frame = original_frame,    #işlenmemiş kare fonksiyona veriliyor
                                                            road_mask_full = road_mask_full     #hesaplama için yol verisi fonksiyona veriliyor.
                                                              )
    #draw_detections return current_frame, best_light_roi, ego_id
    #final_display = kutu ve maskelerin çizildiği final görüntü
    #best_light_roi = en büyük trafik ışığının ham piksel verisini döndürür
    #ego_id şuan ölü değişken durumunda. Kullanılmıyor.
    box_time_1 = time.perf_counter()

    #FPS sayaç bitiş noktası
    t1 = time.perf_counter()
    fps = 1 / (t1 - t0) #1 saniye / zaman farkı


    #FPS ve işlem sürelerini ekrana yazdırma
    cv2.rectangle(final_display, (5, 10), (220, 140), (0, 0, 0), -1)
    cv2.putText(final_display, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(final_display, f"Predict: {(time_predict_1 - time_predict_0)*1000:.1f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(final_display, f"Mask: {(mask_time_1 - mask_time_0)*1000:.1f} ms", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(final_display, f"Box: {(box_time_1 - box_time_0)*1000:.1f} ms", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #final görüntüyü ekrana basma
    cv2.imshow("final_display", final_display)

    #en büyük kırmızı ışığın ekrana getirilmesi
    #en büyük trafik ışığı verisi None değilse ve boyutu 0'dan büyükse True döner.  .size numpy kütüphanesinin bir attribute(öznitelik)'dur. 100 x 50 pikmsel ve 3 kanallı RGB ise 100 x 50 x 3 = 15.000 değerini döndürür.
    if best_light_roi is not None and best_light_roi.size > 0:
        # 1. Mevcut boyutları al (h: yükseklik, w: genişlik)
        h, w = best_light_roi.shape[:2]
        
        # 2. En-Boy oranını hesapla (Genişlik / Yükseklik)
        aspect_ratio = w / h
        
        # 3. Yeni genişliği hesapla (Hedef Yükseklik * Oran)
        # Eğer oran 0.5 ise (dikdörtgen ışık), yeni genişlik 400 * 0.5 = 200 olur.
        new_w = int(400 * aspect_ratio)
        
        # 4. Yeniden boyutlandır (OpenCV'de sıralama: Genişlik, Yükseklik)
        display_roi = cv2.resize(best_light_roi, (new_w, 400))
        
        # 5. Ekrana bas
        cv2.imshow("En Yakin Isik", display_roi)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()