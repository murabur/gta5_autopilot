from PIL import ImageGrab                           #capture screen - PILLOW
import cv2                                          #opencv python
import numpy as np                                  #C tabanlı performanslı liste - matris - tensor işlemleri kütüphanesi
import time                                         #fps hesaplaması için yerleşik zaman kütüphanesi

capture_area = (0, 40, 1280, 760)                   #yakalama bölgesinin koordinatları. sol üst x - sol üst y - sağ alt x - sağ alt y koordinatları. 
while True:
    t0 = time.time()                                #yakalama işleminin başladığı t0 anı

    capture = ImageGrab.grab(bbox= capture_area )   #ImageGrab ile grap işlemi (bbox = capture_area) = yakalanacak görüntünün koordinatları
    frame = np.array(capture)                       #PIL nesnesi numpy arraye çevrilerek opencv'ye verilebilir hale getiriliyor.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #opencv BGR görüntü kabul ediyor. cv2.COLOR_RGB2BGR ile görüntü RGB'den BGR'ye çevriliyor. Bu işlem yapılmadığı vakit olacakları görmek için bu satırı yorum satırına alabilirsiniz. 

    t1 = time.time()                                #yakalama işleminin bittiği t1 anı
    fps = int(1/(t1 - t0))                          # 1 saniye bölü zaman farkı = FPS

    cv2.rectangle(img=frame, pt1=(0, 10), pt2=(180, 70), color=(0, 0, 0), thickness=-1) #yazının altına siyah bant çekme işlemi, istemiyorsanız yorum satırına alın.


    cv2.putText(img=frame, text=f"FPS: {fps}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2) #parametreler kendini açıklıyor. 
    #img = görüntü matrisi, text= yazı #org yazının matristeki pozisyonu #fontface = font , fontscale = yazı boyutu katsayısı , color = BGR - Blue Green Red formatında yazı rengi, thickness = kalınlık



    cv2.imshow("PIL ImageGrab", frame)      

    if cv2.waitKey(1) & 0xFF == ord("q"): #görüntüyü 1ms beklet ve  q'ya basılırsa işlemi gerçekleştir(break: döngüyü(while True) kır)
        break


