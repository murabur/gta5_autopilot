#önemli not: Kod hala geliştirmeye açıktır. Bazı değerlerin iki farklı değişkene atanarak gereksiz kalabalık yapıldığını görebilirsiniz.
#Bu eksiklikleri zaman içerisinde düzeltmeye çalışacağım.  Bazı meseleler kod okunabilirliği adına tercih edilmiştir.

#önemli not2: Kod CPU tarafında ciddi performans optimizasyonuna ihtiyaç duymaktadır.

import bettercam
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# 1. YARDIMCI FONKSİYON: LETTERBOX DÜZELTMESİ (KAYMA ÇÖZÜMÜ)
# ══════════════════════════════════════════════════════════════════════════════
#masks_data = Modelden çıkan ham, bozuk maske verisi(640*640)
#target_h = Gerçek ekranının yüksekliği (720)
#target_w = Gerçek ekranın genişliği (1280)

def strip_letterbox(masks_data, target_h, target_w): 
    """
    .engine letterbox padding'ini (siyah boşlukları) kırpar.
    Böylece maskeler havada uçmaz, asfalta ve araçlara tam oturur.
    """
    #masks_data.shape YOLO'dan gelen maskenin güncel boyutlarını döndürür
    #shape[0] maske sayısı(kaç araba/yol var)
    #shape[1] yükseklik
    #shape[2] genişlik
    #Bizim örneğimizde mask_h = 640, mask_w = 640
    mask_h, mask_w = masks_data.shape[1], masks_data.shape[2]
    
    # Sadece maske kare ise (örn: 640x640) ve ekranımızla uyuşmuyorsa çalışır
    #eğer maske kareyse VE maske yüksekliği ile hedef yükseklik eşit değilse.
    if mask_h == mask_w and mask_h != target_h:
        scale = min(mask_h / target_h, mask_w / target_w) #min(640/720, 640/1280) yani min(0.88,0.5) en küçük olan 0.5'i seçer
        new_h = int(target_h * scale) #720*0.5 = 360
        new_w = int(target_w * scale) #1280*0.5 = 640
        
        # Eklenen siyah boşluk miktarını bul
        pad_top = (mask_h - new_h) // 2 # (640 - 360)/2 = 140
        pad_left = (mask_w - new_w) // 2 #(640 - 640)/2 = 0
        
        # Siyah boşlukları maskeden makasla kes
        masks_data = masks_data[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] #masks_data[:, 140:140+360, 0:0+640 ] yani masks_data[:,140:500, 0:640]
        mask_h, mask_w = new_h, new_w #mask_h = 360 mask_w = 640
    
    return masks_data, mask_h, mask_w



def get_road_centerline(road_mask):
    #yol maskesinin her satırı için orta nokta hesaplaması yapar

    heigth, width = road_mask.shape #maskenin yükseklik ve genişlik değerleri alınıyor.
    center_points = [ ]             #merkez noktası için boş liste

    for y in range(int(heigth*0.3), heigth, 10): #yüksekliğin 10'da 3'lük kısmından başlıyoruz. Yüksekliğin sonuna kadar 10'ar adımla gidiyoruz.
        row = road_mask[y,:] #"Görüntünün y yüksekliğindeki tüm yatay piksellerini bir şerit olarak alıyoruz.
        white_pixels = np.where(row>0.5)[0] #np.where tuple döndürür. Biz içindeki ilk elemanı(listeyi) alıyoruz.

        if len(white_pixels) > 0:
            center_x = int(np.mean(white_pixels)) #tek boyutlu numpy array içindeki değerlerin ortalaması alınır.
            center_points.append((center_x, y)) #eşleştirilen koordinat çifti listeye tuple şeklinde eklenir.

    return center_points
# ══════════════════════════════════════════════════════════════════════════════
# 2. ANA KURULUM
# ══════════════════════════════════════════════════════════════════════════════

# model ağırlık dosyası
# https://drive.google.com/file/d/1TOzAy7CnA6YrCIa_EtZaS10lQ8-YKc5P/view?usp=sharing



#.engine dosyaları derlendiği donanıma özeldir.Nvidia GPU'nuz varsa mutlaka .pt uzantılı pytorch dosyanızdan onnx formatına ardından .engine TensorRT formatına derlemeyi yapın.
MODEL_PATH = r"YOLO\best.pt"
model = YOLO(MODEL_PATH)

camera = bettercam.create(output_color="BGR")
capture_area = (0, 40, 1280, 760)

CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'traffic_light'}
CLASS_COLORS = {
    0: (255, 0, 255),   # road          -   Mor
    1: (0, 255, 255),   # sidewalk      -   Sarı
    2: (255, 0, 0),     # car           -   Mavi
    3: (0, 165, 255),   # motorcycle    -   Turuncu
    4: (0, 255, 0),     # person        -   Yeşil
    5: (0, 0, 255)      # traffic_light -   Kırmızı
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANA DÖNGÜ 
# ══════════════════════════════════════════════════════════════════════════════

def screen_capture(cam_obj, area):
    frame = cam_obj.grab(region=area)
    
    if frame is None:
        return None
    return frame

while True:
    t0 = time.perf_counter()
    frame = screen_capture(camera, capture_area)
    if frame is None: continue

    results = model.predict(source=frame, conf=0.3, verbose=False, half=True)[0]

    if frame is not None:
        annotated_frame = frame.copy()
        overlay = frame.copy() 


    target_h, target_w = frame.shape[:2] # Bizim ekranın boyutu (720, 1280)

    # --- MASKE BOYAMA (SEGMENTASYON) ---
    if results.masks is not None:
            
            # 1. SENARYO: TensorRT (.engine) -> Matris (Data) ve Manuel Kırpma
            if MODEL_PATH.endswith(".engine"):
                raw_masks = results.masks.data.cpu().numpy()
                classes_for_masks = results.boxes.cls.cpu().numpy().astype(int)
                
                stripped_masks, mask_h, mask_w = strip_letterbox(raw_masks, target_h, target_w)
                
                for i, mask in enumerate(stripped_masks):
                    class_id = classes_for_masks[i]
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    overlay[mask_resized > 0.5] = color

            # 2. SENARYO: PyTorch (.pt) -> Poligon (XY) ve Otomatik Hizalama
            elif MODEL_PATH.endswith(".pt"):
                masks_xy = results.masks.xy
                classes_for_masks = results.boxes.cls.cpu().numpy().astype(int)
                
                for i, mask_pts in enumerate(masks_xy):
                    if len(mask_pts) == 0: # Boş maske hatasını önle
                        continue
                    class_id = classes_for_masks[i]
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))


                    if class_id == 0 or class_id == 1:
                        pts = np.array(mask_pts, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color)
                        if class_id == 0:
                            temp_road_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                            cv2.fillPoly(temp_road_mask, [pts], 255)
                            center_pts = get_road_centerline(temp_road_mask)
                            #bu fonksiyon içinde tuple olan liste döndürüyor.


                            if len(center_pts) > 1:
                                for i in range(len(center_pts) - 1 ):
                                    cv2.line(annotated_frame, center_pts[i], center_pts[i+1], (0, 255, 255), 2)
                                    print(f"{center_pts[i]} - {center_pts[i+1]}")


            # HER İKİ SENARYO İÇİN ORTAK: Üst üste bindirme işlemi
            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

    # --- KUTU ÇİZİMİ (DETECTION) ---
    if results.boxes is not None: #eğer sonuç None dönmüyorsa 
        boxes = results.boxes.xyxy.cpu().numpy().astype(int) #YOLO results'dan boxların x,y koordinatlarını CPU'ya ve numpy'a int formatında indir.
        classes = results.boxes.cls.cpu().numpy().astype(int) #YOLO results'dan boxların sınıflarını CPU'ya ve numpy'a int formatında indir
        confidences = results.boxes.conf.cpu().numpy() #güven skoru(confidence) değerlerini YOLO results'dan CPU ve numpy'a indir.(int formatında değil çünkü güven skorları 0 - 1 arası)

        #ekrandaki en büyük bounding boxa sahip trafik ışığını bulma
        max_area = 0
        best_light_roi = None 


        for i, box in enumerate(boxes): #boxes adlı değişkenin elemanlarını numaralandır ve indeksleri i'ye ata
            x1, y1, x2, y2 = box #boxes değişkeninden dönen koordinatları sırasıyla x1, y1, x2, y2 değişkenlerine ata
            class_id = classes[i] #classes değişkenine sınıf numaraları atanmıştı. index ile sınıf numarasını çek ve class_id'ye ata
            conf = confidences[i] #confidences değişkenine confindence(güven skoru) değerleri atanmıştı. Index ile confidence değerini çek ve conf değişkenine ata
            name = CLASS_NAMES.get(class_id, "Bilinmeyen")
            #Dinamik trafik ışığı bulma
            if name == "traffic_light":
                current_area = (x2 - x1) * (y2 - y1)

                #eğer bu ışık hafızadakinden daha büyükse
                if current_area > max_area:
                    max_area = current_area
                    best_light_roi = frame[y1:y2, x1:x2].copy()
                    


            #ÖNEMLİ NOT: "CTRL + fare sol tık"ı ile CLASS_COLORS ve CLASS_NAMES ifadelerine tıklayarak bunların tanımlandığı yere gidebilir ve
            #neyin tanımlandığını görebilirsiniz. Bunu görmeniz bu alttakileri anlamınız açısından önemli.
            color = CLASS_COLORS.get(class_id, (0, 255, 0)) #CLASS_COLORS sözlüğünden class_id'ye göre sınıf rengini al 
            name = CLASS_NAMES.get(class_id, "Bilinmeyen")  #CLASS_NAMES sözlüğünden class_id'ye göre sınıf ismini al. 

            if name != "road" and name != "sidewalk":
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                #sınıf isimlerini yazıyor
                cv2.putText(annotated_frame, f"ID:{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame = annotated_frame 

    # --- FPS VE EKRAN ---
    t1 = time.perf_counter()
    fps = 1 / (t1-t0)


    #f string .1f virgülden sonra bir basamak al.
    cv2.rectangle(frame, (5, 20), (170, 60), (0, 0, 0), -1)
    cv2.putText(img=frame, text=f"FPS: {fps:.1f}", org=(10, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("GTA 5 otopilot", frame)

    if best_light_roi is not None:
        display_roi = cv2.resize(best_light_roi, (200,400))
        cv2.imshow("En Yakin Isik", display_roi)
    else:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()