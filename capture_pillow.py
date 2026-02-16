from PIL import ImageGrab  #capture screen - PILLOW
import cv2 #opencv python
import numpy as np 
import time

capture_area = (0, 40, 1280, 760
                )
while True:
    fps1 = time.time()

    capture = ImageGrab.grab(bbox= capture_area )
    frame = np.array(capture)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps2 = time.time()
    fps = int(1/(fps2 - fps1))
    cv2.putText(frame, f"FPS: {fps}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


