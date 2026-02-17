import mss
import cv2
import numpy as np
import time

# MSS'de "monitor" tanımı sözlük olarak yapılır, bu doğru.
monitor = {"top": 40, "left": 0, "width": 1280, "height": 720}

# Sadece bir kez başlat (Context manager 'with' döngü dışında daha hızlı olabilir)
sct = mss.mss()

counter = 0
fps_text = "0"
start_time = time.time()

print("MSS Test Başlatılıyor...")

while True:
    # 1. Görüntü Yakalama (Bu işlem CPU'da yapılır ve yavaştır)
    # MSS raw bytes döner, bunu numpy array'e çevirmek maliyetlidir.
    img = sct.grab(monitor)
    
    # 2. Dönüştürme İşlemleri
    frame = np.array(img)
    
    # MSS BGRA (4 kanal) döner, OpenCV BGR (3 kanal) ister.
    # Bu dönüşüm işlemciyi yorar ama mecburuz.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 3. FPS Hesaplama (Senin sevdiğin sayaç mantığı)
    counter += 1
    if counter >= 30: # 5 yerine 30 yapalım, işlemci nefes alsın
        end_time = time.time()
        fps = counter / (end_time - start_time)
        fps_text = f"{int(fps)}"
        
        counter = 0
        start_time = time.time() # Süreyi sıfırla

    # 4. Görselleştirme
    cv2.putText(frame, f"FPS: {fps_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("MSS Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()