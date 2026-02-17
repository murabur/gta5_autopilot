import bettercam
import cv2
import numpy as np
import time

capture_area = (0, 40, 1280, 760) 
camera = bettercam.create(output_color="BGR")

counter = 0
# 1. Değişkeni string ("0") değil, sayı (0) olarak başlatıyoruz
display_fps = 0 

while True:
    loop_start = time.time()
    
    capture = camera.grab(region=capture_area)
    if capture is None: continue
    
    process_time = time.time() - loop_start
    
    if process_time > 0:
        current_fps = 1.0 / process_time
    else:
        current_fps = 0

    counter += 1
    
    # 2. Gereksiz str() dönüşümü kalktı. Sadece sayıyı güncelliyoruz.
    if counter >= 30: 
        display_fps = int(current_fps)
        counter = 0

    # 3. f-string zaten "{display_fps}" kısmını otomatik string yapar.
    cv2.putText(capture, f"FPS: {display_fps}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
    
    cv2.imshow("Frame", capture)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break
   
cv2.destroyAllWindows()