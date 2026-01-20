# PathFinderAI
This research project is to see if the current state of technology, and machine learning algorithms, are sufficient in their ability to automate simple tasks and be more efficient with our missions in SAR and our data collection capabilities.

Annotating images to be able to identify where a track-line is:
<img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/e4366c7e-8e7f-43fa-9b7a-ae8348fb8e76" />


An example of the model learning to being able to identify "track lines".
<img width="661" height="665" alt="image" src="https://github.com/user-attachments/assets/c9867010-dd20-498f-a25c-aeb1a0372cc0" />

Here is the command to train the model: ``bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640``
Here is the command to run the model: `` bash 
yolo task=detect mode=predict model='runs/detect/train/weights/best.pt' source='path/to/some/new_image.jpg'``
