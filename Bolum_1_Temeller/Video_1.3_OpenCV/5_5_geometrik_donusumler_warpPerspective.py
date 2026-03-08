import cv2
import numpy as np

# Tıkladığımız noktaları tutacak havuz
secilen_noktalar = []

def fare_tiklamasi(event, x, y, flags, param):
    global secilen_noktalar, clone_img
    
    # Sadece sol tıka basıldığında işlem yap
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(secilen_noktalar) < 4:
            secilen_noktalar.append([x, y])
            
            # Tıklanan yere kırmızı bir nokta koy
            cv2.circle(clone_img, (x, y), 5, (0, 0, 255), -1)
            
            # Hangi noktayı seçtiğini ekrana yaz
            siralar = ["1. Sol Ust", "2. Sag Ust", "3. Sol Alt", "4. Sag Alt"]
            cv2.putText(clone_img, siralar[len(secilen_noktalar)-1], (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow("1. Noktalari Sirayla Tikla", clone_img)
            
            # 4 nokta tamamlandığında işlemi tetikle
            if len(secilen_noktalar) == 4:
                kus_bakisini_uygula()

def kus_bakisini_uygula():
    global secilen_noktalar, img
    height, width = img.shape[:2]
    
    # Listeyi numpy dizisine çevir
    src_points = np.float32(secilen_noktalar)
    
    # Hedef noktalar (Sırası: Sol Üst, Sağ Üst, Sol Alt, Sağ Alt)
    dst_points = np.float32([
        [0, 0],              
        [width, 0],          
        [0, height],         
        [width, height]      
    ])
    
    # Kum saati olmaması için çizim sırasını düzeltip ekrana yeşil yamuk çiziyoruz
    cizim_sirasi = np.array([secilen_noktalar[0], secilen_noktalar[1], 
                             secilen_noktalar[3], secilen_noktalar[2]], np.int32)
    cv2.polylines(clone_img, [cizim_sirasi], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("1. Noktalari Sirayla Tikla", clone_img)
    
    # Perspektif dönüşümünü uygula
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(img, M, (width, height))
    
    cv2.imshow("2. Mukemmel Kus Bakisi", result)
    
    # Terminale asıl koduna kopyalayacağın değerleri bas
    print("\n--- ---")
    print("src_points = np.float32([")
    print(f"    [{secilen_noktalar[0][0]}, {secilen_noktalar[0][1]}], # Sol Üst")
    print(f"    [{secilen_noktalar[1][0]}, {secilen_noktalar[1][1]}], # Sağ Üst")
    print(f"    [{secilen_noktalar[2][0]}, {secilen_noktalar[2][1]}], # Sol Alt")
    print(f"    [{secilen_noktalar[3][0]}, {secilen_noktalar[3][1]}]  # Sağ Alt")
    print("])")
    print("--------------------------------------------------\n")

# Görüntüyü yükle (Dosya adını kendi fotoğrafına göre düzenle)
img = cv2.imread('Bolum_1_Temeller\Video_1.3_OpenCV\ornek_goruntu_2.tiff') 
if img is None:
    print("Hata: Fotograf bulunamadi!")
    exit()

clone_img = img.copy()

print("Lütfen açılan ekranda şeritlerin üzerine şu sırayla tıklayın:")
print("1 -> Sol Üst \n2 -> Sağ Üst \n3 -> Sol Alt \n4 -> Sağ Alt")

# Pencereyi oluştur ve fareyi dinlemeye başla
cv2.namedWindow("1. Noktalari Sirayla Tikla")
cv2.setMouseCallback("1. Noktalari Sirayla Tikla", fare_tiklamasi)
cv2.imshow("1. Noktalari Sirayla Tikla", clone_img)

cv2.waitKey(0)
cv2.destroyAllWindows()