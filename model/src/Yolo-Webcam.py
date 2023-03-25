from ultralytics import YOLO
import cv2
import cvzone

def convert_to_int(*inputs):
    result = tuple(int(x) for x in inputs)
    return result

cap = cv2.VideoCapture(0)
cap.set(3, 1920) # width
cap.set(4, 1080) # height

model = YOLO("./yolo_weights/yolov8n.pt")

class_names = ["Lioozy", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "iPhone 13 Pro Sierra Blau 256GB",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Turn on Webcam
while True:
    success, img = cap.read()
    results = model(img, stream=True) # stream=True for using generators, more efficient

    for r in results:
        # boundingboxes
        boxes = r.boxes

        for box in boxes:
            # Coordinates for bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = convert_to_int(x1, y1, x2, y2)
            w, h = x2-x1, y2-y1 

            # Create boxes with opencv
            # cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,255), thickness=3)

            # Create boxes with cvzone
            cvzone.cornerRect(img, (x1,y1,w,h), colorR=(255,0,255), colorC=(100,50,250), rt=1, t=2)

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class names
            clf_idx = box.cls[0]
            clf_text = class_names[int(clf_idx)]

            # Add text to bounding box
            cvzone.putTextRect(img, f"{clf_text}:{conf}", (max(0,x1), max(42,y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1=live video

