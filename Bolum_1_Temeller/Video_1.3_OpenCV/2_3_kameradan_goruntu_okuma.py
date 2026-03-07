import cv2
import numpy as np

cap = cv2.VideoCapture(0)               #0. indexteki kamerayı oku ve ata. cap kullanımı standart bir kullanımdır.


#kamera görüntüsü sürekli bir görüntü olduğu için bunu bir döngü içerisinde almamız gerekir.
while True:
    ret, frame = cap.read()             #ret, frame kullanımı standart kullanımlardan biridir. ret ile kameranın okunup okunmadığı bilgisi döndürülür. frame'e görüntü matrisi atanır.
    
    cv2.imshow("Kamera", frame)         #"Kamera" ile pencere ismi belirlenir. sağdaki kısma frame değişkeni yazılarak görüntü matrisi bastırılır.

    
    #görüntüyü 1ms ekranda bekletir. 0 olursa tek bir görüntüyü kapatılasıya dek basar. Yüksek olursa FPS'i sınırlar.  
    #Döngü her saniye 1000 kez dönmeye çalışır, kameranın hızı (FPS) neyse görüntü o hızda akar.
    #& 0xFF== ord("q"): q'ya basılırsa döngüden çık. Klasik opencv yazımıdır. 
    if cv2.waitKey(1) & 0xFF == ord('q'): #Buı 
        break

cap.release() #kamerayı serbest bırakır.
cv2.destroyAllWindows()