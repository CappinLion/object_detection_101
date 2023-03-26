from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np

from sort import *

def convert_to_int(*inputs):
    result = tuple(int(x) for x in inputs)
    return result

cap = cv2.VideoCapture("data/cars.mp4")

# Initializes model with specified weights
model = YOLO("./yolo_weights/yolov8l.pt")

# Check if GPU can be found
if torch.cuda.is_available():
    print(f"Detected GPU. Running on: {torch.cuda.get_device_name()}")
else:
    print("GPU not found. Running on CPU.")


class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "Phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

desired_cls = ["car", "motorbike", "bus", "truck"]

mask = cv2.imread("./data/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) # iou: intersection over union threshold

# Crossing Line
limits = [400, 297, 673, 297] # [x1,y2,x2,y2] in pixels for drawing a line

total_counts = []
number_cars = 0

while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask) # define a certain region to detect on only
    results = model(img_region, stream=True) # stream=True for using generators, more efficient

    detections = np.empty((0,5))

    for r in results:
        # boundingboxes
        boxes = r.boxes

        for box in boxes:
            # Coordinates for bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = convert_to_int(x1, y1, x2, y2)
            w, h = x2-x1, y2-y1     
            
            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class names
            clf_idx = box.cls[0]
            clf_text = class_names[int(clf_idx)]

            if clf_text in desired_cls and conf > 0.3:
                # Create and add text to bounding box
                # cvzone.cornerRect(img, (x1,y1,w,h), colorR=(255,0,255), colorC=(100,50,250), rt=5, t=2, l=7)
                # cvzone.putTextRect(img, f"{clf_text}:{conf}", (max(0,x1), max(42,y1)), scale=0.8, thickness=1, offset=2)
                current_array = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(dets=detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0,0,255), thickness=4)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = convert_to_int(x1, y1, x2, y2, id)
        w, h = x2-x1, y2-y1
        cx, cy = x1+w//2, y1+h//2 # get center points of box edges
        print(result)

        if limits[1]-10<cy<limits[1]+10 and limits[2]>cx and cx>limits[0]:
            if total_counts.count(id) == 0:
                total_counts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0,255,0), thickness=4)
            number_cars = len(total_counts)
        
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f"{id}", (max(0,x1), max(42,y1)), scale=1.5, thickness=2, offset=10)
        cvzone.putTextRect(img, f"car_counter: {number_cars}", (0,50), font=cv2.FONT_HERSHEY_DUPLEX, scale=2)
        cv2.circle(img, (cx, cy), radius=5, color=(0,0,255), thickness=4)

    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1=live video 0=press to move

