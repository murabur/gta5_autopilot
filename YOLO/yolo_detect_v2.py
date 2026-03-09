import numpy as np
import cv2
from ultralytics import YOLO
import bettercam
import time


capture_area = (0, 40, 1280, 760)
camera = bettercam.create(output_color="BGR")


while True:
    start_time = time.perf_counter()


    capture = camera.grab(region=capture_area)
    if capture is None:
        continue

    end_time = time.perf_counter()

    fps = 1 / (end_time - start_time)

    
    cv2.putText(img=capture, text=f"FPS: {fps:.1f}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("capture", capture)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break