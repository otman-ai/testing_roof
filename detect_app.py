from ultralytics import YOLO 
import streamlit as st
import cv2
import numpy as np
st.title("Hi,This is only test web application for extracting the front door from image")
st.write("Please upload image of house with front door ,then the front door extracted from the image will be shown above after seconds.")
st.info("make sure the front door is very clear on the image")
#load the model
@st.cache_resource
def load_model_():
    model = YOLO("train/weights/best.pt")
    return model


model = load_model_()

image_streamlit  = st.file_uploader("Upload your image",type=["jpg","png","jpeg"])

if image_streamlit != None:
    st.image(image_streamlit)
    with open("image.jpg",mode = "wb") as f:
        f.write(image_streamlit.getbuffer())
    img = model.predict(source="image.jpg",
                        stream=True, retina_masks=True)
    for result in img:
        mask = result.masks.cpu().numpy()
        masks = mask.masks.astype(bool)
        ori_img = result.orig_img
        for m in masks:
            new = np.zeros_like(ori_img, dtype=np.uint8)
            new[m] = ori_img[m]

            cv2.imwrite("h.jpg",new)
    st.image("h.jpg")
