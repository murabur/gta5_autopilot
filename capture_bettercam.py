import bettercam                #Desktop Duplication API tabanlı kütüphane. Yalnızca Windows - GPU üzerinden veri çeker
import cv2                      #opencv
import numpy as np              #numpy liste - matris - tensör işlemleri
import time

capture_area = (0, 40, 1280, 760)               #yakalama alanı (Sol Üst X, Sol Üst Y, Sağ Alt X, Sağ Alt Y)
camera = bettercam.create(output_color="BGR")   #create fonksiyonu ile parametreye göre en uygun instance oluşturulur. 
#opencv için BGR modunda çıktı alınarak performans cv2.cvtColor performans kaybı önlenir.


counter = 0         #FPS sayacı tanımlama
fps = 0             #cv2.putText'te hata vermemesi baştan için tanımlanıyor.
t0 = time.time()    #başlangıç zamanı


while True:
    
    capture = camera.grab(region=capture_area)                  #belirlenen alandan ekran görüntüsü numpy array olarak alınır.
    if capture is None: continue                                #bettercam bazen none döndürüp programı çökertebilir. Capture None dönerse "devam et"
    

    #saniye bazlı fps ölçümü
    counter += 1
    t1 = time.time()                                            #bitiş zamanı
    elapsed_time = t1 - t0                                      #geçen zaman
    if elapsed_time >= 1.0:                                     #1 saniye geçtiyse True döndür
        fps = counter
        counter = 0
        t0 = t1                                                 #bitiş zamanını devralma, mikro kaymaların birikerek hata oluşturmasını önler
    
    
    cv2.rectangle(img= capture, pt1=(0, 10), pt2=(180, 70), color=(0, 0, 0), thickness=-1) #yazının altına siyah bant çekme işlemi, istemiyorsanız yorum satırına alın.
    cv2.putText(img=capture, text=f"FPS: {fps}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("Bettercam", capture)

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
    if cv2.getWindowProperty("Bettercam", cv2.WND_PROP_VISIBLE) < 1: #çarpıya basılınca arka planda hayalet şekilde işlemin devam etmesini önler
        break
   
cv2.destroyAllWindows()