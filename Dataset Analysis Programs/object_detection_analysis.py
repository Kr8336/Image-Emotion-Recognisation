import cv2
from skimage import io
import numpy as np
import os
import collections

# Path to dataset
img_category_dir = "D:/Uni UK/Msc Artificial Intelligence/Semester 2/CM316 Applications of Machine Learning/Coursework 2/Flickr"
category = {"amusement":4923, "anger":1255, "awe":3133, "contentment":5356, "disgust":1657, "excitement":2914, "fear":1046, "sadness":2901}

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
           'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load the dataset into img_collection and seperate them by classification
img_collection = {}
for c in category.keys():
        img_collection[f"{c}"] = io.imread_collection(f"{img_category_dir}/{c}/*.jpg")

# Load YOLOv3 model from OpenCV library
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

f = open("object_detection_data.txt", "a")

for c in category.keys():
    # Initialize list to store detected objects
    detected_objects = []
    for img in img_collection[c]:
        # Tranform the image into a blob
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set blob as input
        net.setInput(blob)

        # Retrieve outputs
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Detect objects by going through the outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: #if confidence is greater than 0.5 append object
                    class_name = classes[class_id]
                    detected_objects.append(class_name)

    counts = collections.Counter(detected_objects)
    print(c + ":")
    print(counts)
    f.write(str(c) + ":")
    f.write("\n")
    f.write(str(counts))
    f.write("\n")
