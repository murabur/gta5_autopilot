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

    print(f"FPS {fps:.1f}")

    cv2.imshow("capture", capture)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break