"""
GTA5 Autopilot - 3. Şahıs Fine-tune Veri Toplama ve Otomatik Etiketleme
=========================================================================
Bu script:
1. Belirli aralıklarla ekran görüntüsü kaydeder
2. Mevcut YOLO modeli ile otomatik segmentasyon etiketlemesi yapar
3. YOLO .txt formatında etiket dosyası oluşturur
4. X-AnyLabeling uyumlu JSON dosyası oluşturur
5. Her 4 görüntüden birini val klasörüne atar (75/25 split)

Kullanım:
    python capture_dataset.py

Çıktı yapısı:
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── frame_00001.jpg
    │   │   ├── frame_00001.json    (X-AnyLabeling)
    │   │   └── ...
    │   └── val/
    │       ├── frame_00004.jpg
    │       ├── frame_00004.json    (X-AnyLabeling)
    │       └── ...
    ├── labels/
    │   ├── train/
    │   │   ├── frame_00001.txt     (YOLO format)
    │   │   └── ...
    │   └── val/
    │       ├── frame_00004.txt     (YOLO format)
    │       └── ...
    └── dataset.yaml                (YOLO eğitim config)
"""

import bettercam
import cv2
import numpy as np
import os
import json
import time
import base64
from ultralytics import YOLO
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════════════════════════════════════════

# Model
MODEL_PATH = r"YOLO\best.engine"          # Mevcut model (1. şahıs eğitimli)
TASK = "segment"

# Ekran yakalama
CAPTURE_AREA = (0, 40, 1280, 760)         # Sol üst X, Sol üst Y, Sağ alt X, Sağ alt Y

# Veri toplama ayarları
CAPTURE_INTERVAL = 2.0                     # Kaç saniyede bir görüntü al (saniye)
MAX_IMAGES = 1000                          # Maksimum kaç görüntü toplanacak
CONFIDENCE_THRESHOLD = 0.25                # Düşük eşik — 3. şahıs tespitlerini kaçırmamak için
VAL_EVERY_N = 4                            # Her N görüntüden biri val'e gider (4 = %25 val)

# Sınıflar
CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'traffic_light'}
NUM_CLASSES = len(CLASS_NAMES)

# Çıktı dizini
OUTPUT_DIR = r"C:\Users\muratburc\Desktop\gta5_projesi\veri_seti\dataset_3_sahıs"
IMG_W = 1280                               # Yakalanan görüntü genişliği
IMG_H = 720                                # Yakalanan görüntü yüksekliği

# ══════════════════════════════════════════════════════════════════════════════
# KLASÖR YAPISI OLUŞTURMA
# ══════════════════════════════════════════════════════════════════════════════

def create_directory_structure(base_dir):
    """YOLO eğitim formatına uygun klasör yapısı oluşturur"""
    dirs = [
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] Klasör yapısı oluşturuldu: {base_dir}/")
    return dirs

# ══════════════════════════════════════════════════════════════════════════════
# YOLO TXT FORMATI
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_yolo_seg_txt(masks_xy, classes, img_w, img_h):
    """
    Segmentasyon maskelerini YOLO .txt formatına çevirir.
    Her satır: class_id x1 y1 x2 y2 x3 y3 ... (normalize edilmiş)
    """
    lines = []
    for i, pts in enumerate(masks_xy):
        if len(pts) == 0:
            continue
        class_id = int(classes[i])
        
        # Koordinatları normalize et (0-1 arasına)
        pts_array = np.array(pts, dtype=np.float32)
        pts_normalized = pts_array.copy()
        pts_normalized[:, 0] /= img_w
        pts_normalized[:, 1] /= img_h
        
        # Sınır kontrolü
        pts_normalized = np.clip(pts_normalized, 0.0, 1.0)
        
        # YOLO format: class_id x1 y1 x2 y2 ...
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts_normalized])
        lines.append(f"{class_id} {coords}")
    
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# X-ANYLABELING JSON FORMATI
# ══════════════════════════════════════════════════════════════════════════════

def create_xanylabeling_json(filename, img_w, img_h, masks_xy, classes, confidences):
    """
    X-AnyLabeling uyumlu JSON dosyası oluşturur.
    Bu format ile etiketleri X-AnyLabeling'de açıp düzeltebilirsin.
    """
    shapes = []
    for i, pts in enumerate(masks_xy):
        if len(pts) == 0:
            continue
        class_id = int(classes[i])
        conf = float(confidences[i])
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        
        # Noktaları listeye çevir
        points = [[float(x), float(y)] for x, y in pts]
        
        shape = {
            "label": class_name,
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {
                "confidence": round(conf, 4)
            }
        }
        shapes.append(shape)
    
    json_data = {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": None,         # None = dosya boyutunu küçük tutar
        "imageHeight": img_h,
        "imageWidth": img_w
    }
    
    return json_data

# ══════════════════════════════════════════════════════════════════════════════
# DATASET.YAML OLUŞTURMA
# ══════════════════════════════════════════════════════════════════════════════

def create_dataset_yaml(base_dir):
    """YOLO eğitimi için dataset.yaml dosyası oluşturur"""
    yaml_content = f"""# GTA5 Autopilot - 3. Şahıs Fine-tune Dataset
# Otomatik oluşturuldu: {datetime.now().strftime('%Y-%m-%d %H:%M')}

path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

# Sınıflar
names:
  0: road
  1: sidewalk
  2: car
  3: motorcycle
  4: person
  5: traffic_light

nc: {NUM_CLASSES}
"""
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[INFO] dataset.yaml oluşturuldu: {yaml_path}")
    return yaml_path

# ══════════════════════════════════════════════════════════════════════════════
# ANA VERİ TOPLAMA FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Klasör yapısını oluştur
    create_directory_structure(OUTPUT_DIR)
    create_dataset_yaml(OUTPUT_DIR)
    
    # Model yükle
    print(f"[INFO] Model yükleniyor: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task=TASK)
    
    # Kamera başlat
    camera = bettercam.create(output_color="BGR")
    
    print(f"\n{'='*60}")
    print(f"  GTA5 3. Şahıs Veri Toplama Başlıyor")
    print(f"  Aralık: {CAPTURE_INTERVAL} saniye")
    print(f"  Hedef: {MAX_IMAGES} görüntü")
    print(f"  Val oranı: her {VAL_EVERY_N} görüntüden 1'i")
    print(f"  Confidence eşiği: {CONFIDENCE_THRESHOLD}")
    print(f"  Durdurmak için: Q tuşu veya Ctrl+C")
    print(f"{'='*60}\n")
    
    frame_count = 0
    saved_count = 0
    last_capture_time = 0
    
    try:
        while saved_count < MAX_IMAGES:
            # Ekranı yakala (FPS için sürekli)
            frame = camera.grab(region=CAPTURE_AREA)
            if frame is None:
                continue
            
            current_time = time.time()
            
            # Aralık kontrolü
            if current_time - last_capture_time < CAPTURE_INTERVAL:
                # Canlı önizleme göster
                cv2.imshow("Veri Toplama - Q ile dur", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Kullanıcı durdurdu.")
                    break
                continue
            
            last_capture_time = current_time
            frame_count += 1
            
            # YOLO ile tahmin
            results = model.predict(
                source=frame, 
                conf=CONFIDENCE_THRESHOLD, 
                verbose=False, 
                half=True
            )[0]
            
            # Maske var mı kontrol
            if results.masks is None or len(results.masks.xy) == 0:
                print(f"  [SKIP] Frame {frame_count}: Tespit yok, atlanıyor")
                continue
            
            # Verileri çıkar
            masks_xy = results.masks.xy
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            saved_count += 1
            
            # Train mi val mi?
            if saved_count % VAL_EVERY_N == 0:
                split = "val"
            else:
                split = "train"
            
            # Dosya isimleri
            fname = f"frame_{saved_count:05d}"
            img_filename = f"{fname}.jpg"
            txt_filename = f"{fname}.txt"
            json_filename = f"{fname}.json"
            
            # Yollar
            img_path = os.path.join(OUTPUT_DIR, "images", split, img_filename)
            txt_path = os.path.join(OUTPUT_DIR, "labels", split, txt_filename)
            json_path = os.path.join(OUTPUT_DIR, "images", split, json_filename)
            
            # 1. Görüntüyü kaydet
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 2. YOLO .txt etiketini kaydet
            yolo_txt = mask_to_yolo_seg_txt(masks_xy, classes, IMG_W, IMG_H)
            with open(txt_path, "w") as f:
                f.write(yolo_txt)
            
            # 3. X-AnyLabeling JSON kaydet
            json_data = create_xanylabeling_json(img_filename, IMG_W, IMG_H, masks_xy, classes, confidences)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Bilgi yazdır
            num_detections = len(masks_xy)
            class_counts = {}
            for c in classes:
                name = CLASS_NAMES.get(int(c), "?")
                class_counts[name] = class_counts.get(name, 0) + 1
            
            class_str = " | ".join([f"{k}:{v}" for k, v in class_counts.items()])
            print(f"  [{split.upper():5s}] #{saved_count:4d} | {num_detections:2d} tespit | {class_str}")
            
            # Önizleme: tespitleri göster
            preview = frame.copy()
            for i, pts in enumerate(masks_xy):
                if len(pts) == 0:
                    continue
                class_id = int(classes[i])
                color = [(255,0,255), (0,255,255), (255,0,0), (0,165,255), (0,255,0), (0,0,255)][class_id % 6]
                pts_draw = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [pts_draw], True, color, 2)
            
            # Bilgi paneli
            cv2.rectangle(preview, (5, 5), (350, 80), (0, 0, 0), -1)
            cv2.putText(preview, f"Kaydedilen: {saved_count}/{MAX_IMAGES}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview, f"Split: {split} | Tespit: {num_detections}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Veri Toplama - Q ile dur", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Kullanıcı durdurdu.")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C ile durduruldu.")
    
    finally:
        cv2.destroyAllWindows()
        
        # Özet
        train_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "train"))) // 2  # jpg + json
        val_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "val"))) // 2
        
        print(f"\n{'='*60}")
        print(f"  Veri Toplama Tamamlandı!")
        print(f"  Toplam: {saved_count} görüntü")
        print(f"  Train: {train_imgs} | Val: {val_imgs}")
        print(f"  Dizin: {os.path.abspath(OUTPUT_DIR)}")
        print(f"")
        print(f"  Sonraki adımlar:")
        print(f"  1. X-AnyLabeling ile JSON dosyalarını açıp düzelt")
        print(f"  2. Fine-tune başlat:")
        print(f"     model = YOLO('best.pt')")
        print(f"     model.train(data='{OUTPUT_DIR}/dataset.yaml', epochs=30, imgsz=640)")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()