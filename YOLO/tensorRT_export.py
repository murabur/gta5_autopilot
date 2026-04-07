from ultralytics import YOLO
model = YOLO(r"YOLO\best.pt")
model.export(format="engine", imgsz=640, half=True)