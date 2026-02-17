from PIL import ImageGrab
import numpy as np
import cv2
import bettercam
import time



yakalama_alani = (0, 40, 1280, 760) #sol_x üst_y sağ_x alt_y

camera = bettercam.create()

#YOLO sınıf filtreleme 2: Araba, 3: Motosiklet, 5: Otobüs, 7: Kamyon, 9: Trafik Işığı
#secili_siniflar = [2, 3, 5, 7, 9]

sayac = 0
while True:
    zaman = time.time()

    #görüntü yakalama
    capture = camera.grab(region=yakalama_alani)

    if capture is None:
        print(capture)
        continue

    frame = np.array(capture)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    annotated_frame = frame

    #FPS işlemleri
    zaman2 = time.time()
    fps = 1/(zaman2 - zaman)
    fps = int(fps)
    cv2.putText(annotated_frame, f"FPS: {fps}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
    
    #yatay ve dikey çizgi
    cv2.line(annotated_frame, (640, 0), (640, 720), (0, 255, 255), 2)
    cv2.line(annotated_frame, (0, 360), (1280, 360), (0, 255, 255), 2)
    cv2.imshow("ekran", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Program kullanıcı tarafından kapatıldı")
        break
    if cv2.getWindowProperty("ekran", cv2.WND_PROP_VISIBLE) < 1:
        break
   
cv2.destroyAllWindows()