from ultralytics import YOLO
import cv2

model = YOLO("./yolo_weights/yolov8l.pt") #path to weights
results = model("images/3.png", show=True)

# wait for users input
cv2.waitKey(0)

