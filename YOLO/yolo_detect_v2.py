import bettercam
import cv2
import time
import numpy as np
from ultralytics import YOLO

prev_center_pts = None

# ══════════════════════════════════════════════════════════════════════════════
# 1. YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════
def correct_polygon_padding(masks_xy, target_h, target_w, model_img_size=640):
    """
    Poligon (XY) koordinatlarındaki letterbox padding kaymasını vektörel olarak düzeltir.
    """
    #target_h = 720 - target_w = 1280
    #640/720 - 640/1280 = 
    scale = min(model_img_size / target_h, model_img_size / target_w) #scale = 640/1280=  0,5
    pad_y = (model_img_size - target_h * scale) / 2 #640 - 720*0.5 = 280 -> 280/2 = 140. Alttan üstten 140 pixel paddind
    pad_x = (model_img_size - target_w * scale) / 2 #640 - 1280*0.5 = 0

    corrected_masks = []
    for pts in masks_xy:
        if len(pts) == 0: 
            corrected_masks.append(pts)
            continue
            
        pts_array = np.array(pts, dtype=np.float32)
        pts_array[:, 0] = (pts_array[:, 0] - pad_x) / scale
        pts_array[:, 1] = (pts_array[:, 1] - pad_y) / scale
        
        corrected_masks.append(pts_array.astype(np.int32))
        
    return corrected_masks

#yolun orta noktasını alır, bir listeye ekler ve listeyi döndürür.
def get_road_centerline(road_mask):
    height, width = road_mask.shape
    center_points = []
    
    for y in range(int(height * 0.3), height, 10):
        row = road_mask[y, :]
        white_pixels = np.where(row > 0.5)[0]
        
        if len(white_pixels) > 50:  # yeterli piksel varsa
            left_edge = white_pixels[0]
            right_edge = white_pixels[-1]
            
            # Yolun tamamı değil, sadece sol yarısının ortası (kendi şeridin)
            road_width = right_edge - left_edge
            lane_center = left_edge + road_width // 4  # sağdan süren ülke için
            
            center_points.append((lane_center, y))
    
    return center_points



def detect_lane_lines(frame, road_mask_full):
    """
    Road maskesi içinde gerçek şerit çizgilerini tespit eder.
    frame: orijinal görüntü (BGR)
    road_mask_full: 720x1280 road maskesi (binary)
    """
    # Sadece road alanını maskele
    masked = cv2.bitwise_and(frame, frame, mask=road_mask_full)
    
    # Beyaz ve sarı çizgileri bul
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    
    # Beyaz çizgiler
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Sarı çizgiler
    yellow_lower = np.array([15, 80, 150])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # İkisini birleştir
    line_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Gürültü temizle
    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
    
    # Canny + HoughLinesP
    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=50)
    
    if lines is None:
        return None, None
    
    # Çizgileri sol ve sağ olarak ayır
    left_lines = []
    right_lines = []
    img_center = 640  # 1280/2
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        
        # Yatay çizgileri eleme
        if abs(slope) < 0.3: continue
        
        mid_x = (x1 + x2) / 2
        if mid_x < img_center and slope < 0:
            left_lines.append((x1, y1, x2, y2))
        elif mid_x > img_center and slope > 0:
            right_lines.append((x1, y1, x2, y2))
    
    return left_lines, right_lines


def get_lane_center_from_lines(left_lines, right_lines, img_h):
    """Sol ve sağ şerit çizgilerinin ortasından centerline üretir"""
    center_points = []
    
    # Sol çizgilerin ortalama x pozisyonunu her y için hesapla
    left_xs = []
    for x1, y1, x2, y2 in left_lines:
        left_xs.extend([(x1, y1), (x2, y2)])
    
    right_xs = []
    for x1, y1, x2, y2 in right_lines:
        right_xs.extend([(x1, y1), (x2, y2)])
    
    for y in range(int(img_h * 0.3), img_h, 10):
        # En yakın sol ve sağ noktayı bul
        left_x = None
        right_x = None
        
        closest_left = min(left_xs, key=lambda p: abs(p[1] - y), default=None)
        closest_right = min(right_xs, key=lambda p: abs(p[1] - y), default=None)
        
        if closest_left and closest_right:
            center_x = (closest_left[0] + closest_right[0]) // 2
            center_points.append((center_x, y))
    
    return center_points


def smooth_centerline_polyfit(center_points, target_h):
    if len(center_points) < 3:
        return center_points
    
    xs = np.array([p[0] for p in center_points])
    ys = np.array([p[1] for p in center_points])
    
    # 2. derece polinom uydur (y'ye göre x)
    coeffs = np.polyfit(ys, xs, 2)
    
    # Düzgün noktalar üret
    smooth_ys = np.arange(int(target_h * 0.3), target_h, 10)
    smooth_xs = np.polyval(coeffs, smooth_ys).astype(int)
    
    return list(zip(smooth_xs, smooth_ys))
# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL VE YAPILANDIRMA
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = r"YOLO\best.engine"
model = YOLO(MODEL_PATH, task="segment")

#camera işlemleri
camera = bettercam.create(output_color="BGR")
capture_area = (0, 40, 1280, 760)

CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'traffic_light'}
CLASS_COLORS = {
    0: (255, 0, 255),
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 165, 255),
    4: (0, 255, 0),
    5: (0, 0, 255)
}

target_h = 720
target_w = 1280
small_h = 180
small_w = 320
scale_y = small_h / target_h
scale_x = small_w / target_w
global_overlay = np.zeros((small_h, small_w, 3), dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════════════════
# 3. İŞLEM FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════


#ekran yakalama fonksiyonu
def screen_capture(cam_obj, area):
    frame = cam_obj.grab(region=area)
    if frame is None:
        return None
    return frame

#tahmin fonksiyonu
def get_predictions(source):
    results = model.predict(source=source, conf=0.3, verbose=False, half=True, stream=True)
    return next(results)



#maske işlemleri
def process_lane_data(results, target_h, target_w, annotated_frame, overlay, original_frame):
    global prev_center_pts

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
            # Küçük boyutta road maskesi
            temp_road = np.zeros((small_h, small_w), dtype=np.uint8)
            cv2.fillPoly(temp_road, [best_road_mask], 255)

            # Tam boyut road maskesi (şerit tespiti için)
            road_mask_full = cv2.resize(temp_road, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Şerit çizgilerini bul
            left_lines, right_lines = detect_lane_lines(original_frame, road_mask_full)

            if left_lines and right_lines:
                # Çizgilerden centerline
                current_center_pts = get_lane_center_from_lines(left_lines, right_lines, target_h)
            else:
                # Fallback: maske ortalaması
                current_center_pts = get_road_centerline(temp_road)
                current_center_pts = [(int(x / scale_x), int(y / scale_y)) for x, y in current_center_pts]


            current_center_pts = smooth_centerline_polyfit(current_center_pts, target_h)

            # Temporal smoothing
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
                    cv2.line(annotated_frame, current_center_pts[j], current_center_pts[j + 1], (0, 255, 255), 2)

        overlay_resized = cv2.resize(overlay, (target_w, target_h))
        cv2.addWeighted(src1=overlay_resized, alpha=0.4, src2=annotated_frame, beta=0.6, gamma=0, dst=annotated_frame)

    return annotated_frame


def draw_detections(results, current_frame, original_frame):
    best_light_roi = None
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        max_area = 0

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = classes[i]
            conf = confidences[i]
            name = CLASS_NAMES.get(class_id, "Bilinmeyen")

            if name == "traffic_light":
                current_area = (x2 - x1) * (y2 - y1)
                if current_area > max_area:
                    max_area = current_area
                    best_light_roi = original_frame[y1:y2, x1:x2].copy()  # original_frame'den kırp

            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            if name not in ["road", "sidewalk"]:
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(current_frame, f"ID:{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return current_frame, best_light_roi

# ══════════════════════════════════════════════════════════════════════════════
# 4. ANA DÖNGÜ
# ══════════════════════════════════════════════════════════════════════════════


while True:
    t0 = time.perf_counter()
    
    frame = screen_capture(camera, capture_area)
    if frame is None: continue

    # Tahmin
    time_predict_0 = time.perf_counter()
    results = get_predictions(frame)
    time_predict_1 = time.perf_counter()

    annotated_frame = frame
    original_frame = frame.copy()
    global_overlay.fill(0)

    # Maskeleme
    mask_time_0 = time.perf_counter()
    annotated_frame = process_lane_data(results, target_h, target_w, annotated_frame, global_overlay, original_frame)
    mask_time_1 = time.perf_counter()

    # Kutu Çizimi
    box_time_0 = time.perf_counter()
    final_display, best_light_roi = draw_detections(results, annotated_frame, original_frame)
    box_time_1 = time.perf_counter()

    # Metrikler
    t1 = time.perf_counter()
    fps = 1 / (t1 - t0)

    cv2.rectangle(final_display, (5, 10), (220, 140), (0, 0, 0), -1)
    cv2.putText(final_display, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(final_display, f"Predict: {(time_predict_1 - time_predict_0)*1000:.1f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(final_display, f"Mask: {(mask_time_1 - mask_time_0)*1000:.1f} ms", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(final_display, f"Box: {(box_time_1 - box_time_0)*1000:.1f} ms", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("final_display", final_display)

    if best_light_roi is not None and best_light_roi.size > 0:
        display_roi = cv2.resize(best_light_roi, (200, 400))
        cv2.imshow("En Yakin Isik", display_roi)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()