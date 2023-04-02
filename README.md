# Licence Plate Detection Yolov8
Licence plate recognition is a hot topic in computer vision involving object detection/recognition and optical character recognition (OCR) models. In this practice, we trained a small dataset of vehicle licence plate using Yolov8, then extracting the content with PaddleOCR. Our dataset was dowmloaded from kaggle. It is a small dataset of 228 images. The data ratio is 24 images for testing, 22 images for validation and 182 images for training.
The training is done using yolov8n.pt pretrained model and run for 50 epochs.
Only the best weight is used for inference.
#### dataset visualization
![img_dataset](https://user-images.githubusercontent.com/48753146/229334150-051ca6a5-b40f-4fd1-bfbb-c2cfbf5d7186.png)


# Dependencies

 - Ultralytics import YOLO
 - OpenCV
 -Paddleocr
 - Glob
 - OS
 - Matplotlib
 - Pathlib
 - Argparse

# Training Performance
### Confusion Matrix
![confusion_matrix](https://user-images.githubusercontent.com/48753146/229332875-d613aa6b-265b-4697-bc1d-a092a5df7ef4.png)
### R Curve
![R_curve](https://user-images.githubusercontent.com/48753146/229332873-afe1f868-e093-4f77-9276-a3e58fdee23f.png)
### F1 Curve
![F1_curve](https://user-images.githubusercontent.com/48753146/229332876-f4f5e84e-d1b4-4ead-9baa-b2dab6491019.png)
### Labels
![labels](https://user-images.githubusercontent.com/48753146/229332877-617a08f3-702e-4c2f-9c3e-3633aece1504.jpg)
### Labels Correlogram
![labels_correlogram](https://user-images.githubusercontent.com/48753146/229332878-113f6763-4db9-4b88-845f-3eabd764a590.jpg)
### P Curve
![P_curve](https://user-images.githubusercontent.com/48753146/229332879-5527be71-f6ea-4b1e-ab48-db7c00e7bb1e.png)
### PR Curve
![PR_curve](https://user-images.githubusercontent.com/48753146/229332880-7d7554d3-d5b4-42f6-9fa6-44fa678774e7.png)
### Training Last Batch

# Inference
## On images
When inference is run, the model detect the licence plate, extract it and apply an OCR engine for content extraction
### run this command
![inference_command](https://user-images.githubusercontent.com/48753146/229333816-66c4601a-224b-4e19-911b-1c6c991641ec.png)
### Licence extracted
![test_license_plate](https://user-images.githubusercontent.com/48753146/229333691-700431e2-1485-4591-a13b-92271a914b13.jpg)

## On video
Run below command to apply the detection only on video data:
 >python video_license_plate_detection.py

# Note:
 * The model can be imporved by training on a larger dataset.
