from ultralytics import YOLO
import cv2
import cvzone
import torch

def convert_to_int(*inputs):
    result = tuple(int(x) for x in inputs)
    return result

cap = cv2.VideoCapture("data/cars.mp4")

# Initializes model with specified weights
model = YOLO("./yolo_weights/yolov8l.pt")

# Check if GPU can be found
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
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

while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask) # define a certain region to detect on only
    results = model(img_region, stream=True) # stream=True for using generators, more efficient

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
                cvzone.cornerRect(img, (x1,y1,w,h), colorR=(255,0,255), colorC=(100,50,250), rt=1, t=2, l=7)
                cvzone.putTextRect(img, f"{clf_text}:{conf}", (max(0,x1), max(42,y1)), scale=0.8, thickness=1, offset=2)

    cv2.imshow("Image", img)
    cv2.waitKey(0) # 1=live video 0=press to move

