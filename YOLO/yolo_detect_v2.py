import numpy as np
import cv2
import bettercam
import time
from collections import deque

capture_area = (0, 40, 1280, 760)
camera = bettercam.create(output_color="BGR")

fps_deque = deque(maxlen=60)

loop_start = time.perf_counter()

while True:
    capture = camera.grab(region=capture_area)
    if capture is None:
        continue

    loop_end = time.perf_counter()
    delta_time = loop_end - loop_start
    loop_start = loop_end 

    if delta_time > 0:
        fps = 1 / delta_time
        fps_deque.append(fps)
    
    fps_smooth = np.mean(fps_deque) if fps_deque else 0

    cv2.putText(img=capture, text=f"FPS: {fps_smooth:.1f}", org=(10, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                color=(0, 255, 0), thickness=2)
    
    cv2.imshow("capture", capture)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()