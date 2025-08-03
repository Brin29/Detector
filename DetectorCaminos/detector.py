import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
                     
                
model = YOLO('model/best-first.pt')

results = model.predict(
    source=['data/prueba1.jpg', 'data/prueba2.jpg', 'data/prueba3.jpg', 'data/prueba6.jpg', 'data/prueba5.jpg', 'data/prueba4.jpg'],
    conf=0.25,
    save=False,
    task='segment'
)


for result in results:
  img = result.plot()

  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title("YOLOv8 Segmentación")
  plt.axis('off')
  plt.show()