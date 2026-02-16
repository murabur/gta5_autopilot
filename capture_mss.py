import mss
import cv2
import numpy as np
import time

monitor = {
    "top": 40,       
    "left": 0,       
    "width": 1280,   
    "height": 720    
}

start_time = time.time()
counter = 0
avg_fps = 0

with mss.mss() as sct:
    counter = 0 
    while True:

        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        counter += 1
        
        if counter >= 5:
            end_time = time.time()
            frame_time = end_time - start_time
            avg_fps = int(counter / frame_time)

            counter = 0
            start_time = time.time()


        
        cv2.putText(frame, f"FPS: {avg_fps}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    

