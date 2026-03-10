import bettercam
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

def strip_letterbox_tensor(masks_tensor, target_h, target_w): 
    mask_h, mask_w = masks_tensor.shape[1], masks_tensor.shape[2]
    
    if mask_h == mask_w and mask_h != target_h:
        scale = min(mask_h / target_h, mask_w / target_w)
        new_h = int(target_h * scale)
        new_w = int(target_w * scale)
        
        pad_top = (mask_h - new_h) // 2
        pad_left = (mask_w - new_w) // 2
        
        masks_tensor = masks_tensor[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        mask_h, mask_w = new_h, new_w
    
    return masks_tensor, mask_h, mask_w

def get_road_centerline(road_mask):
    height, width = road_mask.shape
    start_y = int(height * 0.3)
    
    sliced_mask = road_mask[start_y::10, :]
    center_points = []

    for i, row in enumerate(sliced_mask):
        white_pixels = np.nonzero(row)[0]
        if white_pixels.size > 0:
            center_x = int(np.mean(white_pixels))
            real_y = start_y + (i * 10)
            center_points.append((center_x, real_y))

    return center_points

MODEL_PATH = r"YOLO\best.pt"
model = YOLO(MODEL_PATH, task="segment")

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

def screen_capture(cam_obj, area):
    frame = cam_obj.grab(region=area)
    if frame is None:
        return None
    return frame

def get_predictions(source):
    results = model.predict(source=source, conf=0.3, verbose=False, half=True, stream=True)
    return next(results)

def process_lane_data(results, target_h, target_w, annotated_frame, overlay):
    if results.masks is not None:
        if MODEL_PATH.endswith(".engine"):
            masks_tensor = results.masks.data
            classes_for_masks = results.boxes.cls.cpu().numpy().astype(int)
            
            stripped_tensor, _, _ = strip_letterbox_tensor(masks_tensor, target_h, target_w)
            
            # GPU üzerinde toplu resize işlemi
            stripped_tensor = stripped_tensor.unsqueeze(1)
            resized_tensor = F.interpolate(stripped_tensor, size=(target_h, target_w), mode="nearest").squeeze(1)
            
            # Tek seferde CPU'ya al
            resized_masks = resized_tensor.cpu().numpy()

            for i, mask in enumerate(resized_masks):
                class_id = classes_for_masks[i]
                
                if class_id in [0, 1, 2]:
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    mask_bool = mask > 0.5
                    overlay[mask_bool] = color
                    
                    if class_id == 0:
                        temp_road_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                        temp_road_mask[mask_bool] = 255
                        center_pts = get_road_centerline(temp_road_mask)
                        
                        if len(center_pts) > 1:
                            for j in range(len(center_pts) - 1):
                                cv2.line(annotated_frame, center_pts[j], center_pts[j+1], (0, 255, 255), 2)

        elif MODEL_PATH.endswith(".pt"):
            masks_xy = results.masks.xy
            classes_for_masks = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, mask_pts in enumerate(masks_xy):
                if len(mask_pts) == 0: continue
                class_id = classes_for_masks[i]
                
                if class_id in [0, 1, 2]:
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    pts = np.array(mask_pts, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    
                    if class_id == 0:
                        temp_road_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                        cv2.fillPoly(temp_road_mask, [pts], 255)
                        center_pts = get_road_centerline(temp_road_mask)

                        if len(center_pts) > 1:
                            for j in range(len(center_pts) - 1):
                                cv2.line(annotated_frame, center_pts[j], center_pts[j+1], (0, 255, 255), 2)

        cv2.addWeighted(src1=overlay, alpha=0.4, src2=annotated_frame, beta=0.6, gamma=0, dst=annotated_frame)

    return annotated_frame

def draw_detections(results, current_frame):
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
                    best_light_roi = current_frame[y1:y2, x1:x2].copy()

            color = CLASS_COLORS.get(class_id, (0, 255, 0))

            if name not in ["road", "sidewalk"]:
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(current_frame, f"ID:{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return current_frame, best_light_roi

target_h = 720
target_w = 1280
global_overlay = np.zeros((target_h, target_w, 3), dtype=np.uint8)

while True:
    t0 = time.perf_counter()
    frame = screen_capture(camera, capture_area)
    if frame is None: continue

    results = get_predictions(frame)

    annotated_frame = frame
    global_overlay.fill(0)

    annotated_frame = process_lane_data(results, target_h, target_w, annotated_frame, global_overlay)
    final_display, best_light_roi = draw_detections(results, annotated_frame)

    t1 = time.perf_counter()
    fps = 1 / (t1 - t0)

    cv2.rectangle(final_display, (5, 20), (170, 60), (0, 0, 0), -1)
    cv2.putText(img=final_display, text=f"FPS: {fps:.1f}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("GTA 5 otopilot", final_display)

    if best_light_roi is not None:
        display_roi = cv2.resize(best_light_roi, (200, 400))
        cv2.imshow("En Yakin Isik", display_roi)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()