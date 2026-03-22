#Bu kod ile GTA 5 üzerinde hareket tespiti yapılıyor.
#cv2.abssdiff.png dosyasında sonuç görülebilir.

import cv2
import numpy as np
import bettercam


capture_area = (0, 40, 1280, 760) 
camera = bettercam.create(output_color="BGR")

capture_eski = None
capture_yeni = None
cv2.namedWindow("Bettercam")

while True:
    capture = camera.grab(region=capture_area)
    if capture is None: continue 
    
    capture_eski = capture_yeni    
    capture_yeni = capture
    
    if capture_eski is not None:

        hareket = cv2.absdiff(capture_eski, capture_yeni)
        cv2.imshow("Bettercam", hareket)    

    
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
    if cv2.getWindowProperty("Bettercam", cv2.WND_PROP_VISIBLE) < 1: #çarpıya basılınca arka planda hayalet şekilde işlemin devam etmesini önler
        break
   
cv2.destroyAllWindows()