import numpy as np
import cv2
import bettercam
from ultralytics import YOLO
import time


model = YOLO(r"YOLO\best.engine", task="segment")
#camera işlemleri
capture_area = (0,40,1280,760)
camera = bettercam.create(output_color="BGR")
camera.start(region= capture_area)
camera.is_capturing


while True:


    frame = camera.get_latest_frame()

    results = model.predict(source = frame, conf = 0.5, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    for i in boxes:
        x1, y1, x2, y2 = i
        cv2.rectangle(frame, pt1= (x1,y1), pt2= (x2,y2), color=(0,255,255), thickness=2 )

    



    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.stop()
camera.is_capturing