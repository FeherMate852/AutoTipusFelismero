from ultralytics import YOLO

# Modell betöltése
model = YOLO('yolov8n.pt') 

# Tanítás
model.train(data='./Car_types_Yolo/data.yaml', epochs=50, imgsz=640)
