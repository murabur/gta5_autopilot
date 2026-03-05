#Seri GTA5 üzerine olacağından ekran kaydı üzerinden görüntü okuyacağız.
#webcam için yapay zekadan kod talep edebilirsiniz. Ayrıca bir örneğini de muhtemelen eklerim.


#Önemli not: Kodu sindirmenin en etkili yolu sadece bir elemanı değiştirerek ne sonuç verdiğini canlı canlı görmektir. 
#capture_area'nın içerisindeki değerlerden biri değiştirilerek sonuç görülebilir.


import bettercam                #Desktop Duplication API tabanlı kütüphane. Yalnızca Windows - GPU üzerinden veri çeker - yüksek pe
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
        fps = counter                                           #fpsi 1 saniyede sayılan kare sayısına - counter eşitliyoruz.
        counter = 0                                             #sayacı sıfırla
        t0 = t1                                                 #bitiş zamanını devralma, mikro kaymaların birikerek hata oluşturmasını önler
    
    

    #opencv dikdörtgen çizdirme komutuı.
    #img=capture işlem yapılacak görüntü
    #pt1 ve pt2 koordinatlar. x = 0 y=10 - x180 y = 70 komutları. Görüntü bir matristir ve x = 0, y = 0 koordinatlarından sol üst taraftan başlar.
    #color = (B,G,R) 0,0,0, siyah görüntü.
    #thickness = -1  şeklin içini tamamen doldur.
    cv2.rectangle(img= capture, pt1=(0, 10), pt2=(180, 70), color=(0, 0, 0), thickness=-1) 


    #cv2.putText opencv yazı yazma komutu
    #img=capture işlem yapılacak görüntü
    #text = yazılacak metin. f string ile sayısal formattaki FPS verimiz text=f"FPS: {fps:.1f}" stringin içine yerleştiriliyor. .1f bir basamak göster.
    #org(10,50) origin, metnin başladığı SOL ALT köşe.
    #fontface = font tipi - fontscale = font büyüklüğü - color = B,G,R formatında renk, thickness=kalınlık
    cv2.putText(img=capture, text=f"FPS: {fps:.1f}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("Bettercam", capture) #görüntüyü ekrana bas. "Bettercam" pencere ismi capture basılacak görüntü

    #görüntüyü 1ms ekranda bekletir. 0 olursa tek bir görüntüyü kapatılasıya dek basar. Yüksek olursa FPS'i sınırlar. 
    #& 0xFF== ord("q"): q'ya basılırsa döngüden çık. Klasik opencv yazımıdır. 
    if cv2.waitKey(1) & 0xFF== ord("q"): 
        break


    #çarpıya basılınca arka planda hayalet şekilde işlemin devam etmesini önler.  
    #Fark edeceğiniz üzere "Bettercam" ifadesi yer alıyor. cv2.imshow ile aynı olmazsa pencereyi kapatır.
    if cv2.getWindowProperty("Bettercam", cv2.WND_PROP_VISIBLE) < 1:
        break
   
cv2.destroyAllWindows() #bütün opencv pencerelerini kapat.