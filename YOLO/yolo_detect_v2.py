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
    results = model.track(source=source, conf=0.3, verbose=False, half=True, persist=True, stream=True)
    return next(results)


ego_track_id = None

def find_ego_car(boxes, classes, track_ids):
    global ego_track_id
    
    if track_ids is None:
        return None
    
    # Mevcut ego hâlâ sahnede mi?
    if ego_track_id is not None:
        if ego_track_id in track_ids:
            return ego_track_id
    
    # Ego kayboldu veya henüz atanmadı — yeniden bul
    best_score = 0
    best_id = None
    
    screen_center_x = 640
    screen_bottom = 720
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if classes[i] != 2:
            continue

  
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        #ekranın üst yarısındaysa atla
        if cy < screen_bottom * 0.5:
            continue
        
        area = (x2 - x1) * (y2 - y1)

        
        dist_x = abs(cx - screen_center_x)
        dist_y = abs(cy - screen_bottom)
        closeness = 1 / (dist_x + dist_y + 1)
        
        score = closeness * closeness * area
        if score > best_score:
            best_score = score
            best_id = track_ids[i]


    
    ego_track_id = best_id
    return ego_track_id

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

def draw_detections(results, current_frame, original_frame, road_mask_full):
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
        print(f"ego_id: {ego_id}, track_ids: {track_ids}")

        max_area = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = classes[i]
            conf = confidences[i]
            name = CLASS_NAMES.get(class_id, "Bilinmeyen")
            tid = track_ids[i] if track_ids is not None else "?"

            if name == "traffic_light":
                current_area = (x2 - x1) * (y2 - y1)
                if current_area > max_area:
                    max_area = current_area
                    best_light_roi = original_frame[y1:y2, x1:x2].copy()

            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            if name not in ["road", "sidewalk"]:
                if track_ids is not None and track_ids[i] == ego_id:
                    # Ego car yolda mı kontrol
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

    return current_frame, best_light_roi, ego_id

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
    annotated_frame, road_mask_full = process_lane_data(results, target_h, target_w, annotated_frame, global_overlay)
    mask_time_1 = time.perf_counter()

    # Kutu Çizimi
    box_time_0 = time.perf_counter()
    final_display, best_light_roi, ego_id = draw_detections(results, annotated_frame, original_frame, road_mask_full)
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