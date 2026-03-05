import cv2

# Kamera yerine video dosyasının yolunu yazıyoruz. kamera olsaydı 0 - 1 gibi index numarasını yazacaktık.
# gerisi aynı
cap = cv2.VideoCapture("video_dosyasi.mp4") #video yolunu kendinize göre ayarlayın. Bu egitimde GTA 5 ekran kayıtları üzerinden gidilecektir. Siz ekran kaydı yerine video dosyalarını kullanabilirsiniz.

while True:
    ret, frame = cap.read()
    
    # Video bittiğinde ret False döner, bu durumda döngüyü kırıyoruz.
    if not ret:
        break

    cv2.imshow("Video Dosyasi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  #burada 30 FPS'lik bir görüntüyü normal hızında oynatmak için waitKey(33) demelisiniz çünkü 1000/33=30 -> 1 saniye/33 ms = 30 Frame Per Second
        break

cap.release()
cv2.destroyAllWindows()