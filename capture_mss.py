import mss                                                      #işletim sistemine göre ekran yakalama kullanır. Windows GDI - Linux X11
import cv2                                                      #opencv
import numpy as np                                              #numpy liste - matris - tensör işlemleri
import time


monitor = {"top": 40, "left": 0, "width": 1280, "height": 720}  #mss kütüphanesi koordinatları sözlük şeklinde alır. anahatarlar ne anlama geldiğini açıklıyor.

sct = mss.mss()                                                 #mss kütüphanesindeki mss class'ından sct adlı bir instance oluşturur.


counter = 0                                                     #FPS sayacı tanımlama
fps = 0                                                         #cv2.putText'te hata vermemesi baştan için tanımlanıyor.
t0 = time.time()                                                #başlangıç zamanı 

while True:

    img = sct.grab(monitor)                                     #belirlenin bölgenin görüntüsünü ham veri olarak çeker.
    frame = np.array(img)                                       #ham veri opencv'ye verilmek üzere numpy array'e çevrilir.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)             #mss BGRA döndürür, opencv BGR kabul eder, cv2.COLOR_BGRA2BGR ile dönüştürme işlemi gerçekleştirilir.

    
    #1 saniye geçtiğinde counter kaç oldu, bu sayede kaç tane frame çizildiği hesaplanır çünkü img=sct.grab ve diğer satırlar bitmeden counter artmaz.
    counter += 1
    t1 = time.time()                                            #bitiş zamanı
    elapsed_time = t1 - t0
    if elapsed_time >= 1.0:
        fps = counter
        counter = 0
        t0 = t1
    

    cv2.rectangle(img=frame, pt1=(0, 10), pt2=(180, 70), color=(0, 0, 0), thickness=-1) #yazının altına siyah bant çekme işlemi, istemiyorsanız yorum satırına alın.
    cv2.putText(img=frame, text=f"FPS: {fps}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("MSS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()