from logging import NullHandler
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"C:\Users\SWETHA KRISHNA\Downloads\best.pt")

st.title('Detection of Fashion Products')

st.markdown('This app detects the clothing items present in any image. It uses the YOLOv5 object detection model.')

choice = st.radio(
     "Select input type",
     ('File upload', 'Image from camera'))

if choice == 'File upload':
  uploaded_file = st.file_uploader("Choose a file",type = 'jpg')
  if uploaded_file is not None:
    img = Image.open(uploaded_file)
    output = yolo_model(img, size = 640)
    with st.container():
      st.markdown('Fashion products and attributes detected:')
      st.image(np.squeeze(output.render()), width = 450)
      class_list = output.pandas().xyxy[0].name.tolist()
      for i in class_list:
        st.write(str(i).capitalize())
     
else:
  uploaded_file = st.camera_input("Take a picture")
  if uploaded_file is not None:
    img = Image.open(uploaded_file)
    output = yolo_model(img, size=640)
    with st.container():
      st.markdown('Fashion products and attributes detected:')
      st.image(np.squeeze(output.render()), width = 450)
      class_list = output.pandas().xyxy[0].name.tolist()
      for i in class_list:
        st.write(str(i).capitalize())
     