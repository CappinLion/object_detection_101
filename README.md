# object_detection_101

This project serves to introduce the basics of object detection and applies them to perform a car counter with yolov8.
In order to track the amount of cars, we are applying SORT() which is a simple online realtime tracking algorithm. (Reference: https://github.com/abewley/sort)
It is highly recommended to utilize GPU the use large yolov8 model.
Lastly, only cars are counted within a certain region. For that we masked the image with `cv2.bitwise_and(img, mask)`. The mask can easily be created with canva.com

## Version
Version 0.1.0 - 24.03.2023

## Contributor
Lion Ly

## Requirements and dependencies
First install all dependencies with `poetry install`.
After that you need to `poe force-cudall` which is a task runner (Poe the Poet) that works well with poetry. This allows you to run any arbitrary command such as pip installations.
We do that in order to use PyTorch wheels repo to utilize CUDA GPU. 
(Reference: https://github.com/nat-n/poethepoet)