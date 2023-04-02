# Libraries Importation
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse



# Verify if the folder used to save extracted licence plate is present, otherwise create it
Path("cropped_license_plates/").mkdir(parents=True, exist_ok=True)

def licence_plate_extraction(model,image):
    # Load a model
    model = YOLO(model)  # load a custom model
    # Predict with the model
    results = model(image, save=True, conf=.3)
    res = list(results)[0] # get result from generator
    
    # get coordinates from yolov8 detection and convert from torch tensor to list
    coord = res.boxes.xyxy
    coord = coord.tolist()
    
    # Clean the folder used for saving previous detected plates
    files = glob.glob('cropped_license_plates/*')
    for f in files:
        os.remove(f)

    # Save the detected license plates
    count = 0
    for i in coord:
        
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[2])
        y2 = int(i[3])

        img = cv2.imread(image)
        roi = img[y1:y2, x1:x2]
        name = f"cropped_license_plates/license_plate_{count}.jpg"
        cv2.imwrite(name, roi)
        count+=1
        
    # PaddleOCR for license place content extraction
    for lp in glob.glob("cropped_license_plates/*.jpg"):
        ocr_model = PaddleOCR(lang='en', show_log=False, use_angle_cls=True, use_gpu=False)
        result = ocr_model.ocr(lp, cls=True)
        for line in result:
            print(f" Licence Plate: {line[1][0]}")


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)

    # Parse the argument
    args = parser.parse_args()
    
    licence_plate_extraction(model=args.model, 
                             image=args.image_path)
