import torch
from PIL import Image
import cv2
import numpy as np
import os
from argparse import ArgumentParser
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def load_model():
    # Load YOLOv5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(model, image_path):
    # Read image
    img = Image.open(image_path)
    
    # Inference
    results = model(img)
    
    # Results
    results.print()  # Print results to console
    
    # Get detected objects
    detected_objects = results.pandas().xyxy[0]
    
    return img, detected_objects

def get_yolo_format_string(x1, y1, x2, y2, height, width):
    #Lemamaesh!
    pass

def label_image(image_name, img, detected_objects, image_label):
    for _, obj in detected_objects.iterrows():
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        label = f"{obj['name']} {obj['confidence']:.2f}"
        if obj['name'].lower().strip() == 'cat':
            #TODO: update to real height!            
            height = 300
            #TODO: update to real height!
            width = 640
            yolo_str = get_yolo_format_string(int(x1), int(y1), int(x2), int(y2), int(height), int(width))
            with open(image_name +'.xml.txt', 'w') as LABEL:
                LABEL.write(yolo_str)
                



def draw_boxes(img, detected_objects):
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for _, obj in detected_objects.iterrows():
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        label = f"{obj['name']} {obj['confidence']:.2f}"
        if obj['name'].lower().strip() == 'cat':
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    return img_cv

def detect_and_box_objects(model, image_path):
    img, detected_objects = detect_objects(model, image_path)
    result_img = draw_boxes(img, detected_objects)
    return result_img
    

def has_cat(detected_objects):
    for _, obj in detected_objects.iterrows():
        label = f"{obj['name']} {obj['confidence']:.2f}"
        if obj['name'].lower() == 'cat':
            return True
    return False
        


def display_image_cv2(img):
    # Display the image using OpenCV
    cv2.imshow('Object Detection Result', img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window when a key is pressed



def main():
    model = load_model()
    image_path = 'path/to/your/640x480_image.jpg'
    result_img = detect_and_box_objects(model, image_path)

    # Save the result
    cv2.imwrite('result.jpg', result_img)
    print("Result saved as 'result.jpg'")

if __name__ == '__main__':
    main()