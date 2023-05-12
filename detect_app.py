from ultralytics import YOLO 
import streamlit as st
import cv2
import numpy as np
from skimage.io import imsave
st.title("Hi,This is only test web application for extracting the Wall from image")
st.write("Made by otman heddouch")
st.write("Please upload image of house with Roof ,then the Wamm extracted from the image will be shown above after seconds.")
st.info("make sure the wall is very clear on the image")
#load the model
@st.cache_resource
def load_model_():
    model = YOLO("wall.pt")
    return model


model = load_model_()

image_streamlit  = st.file_uploader("Upload your image",type=["jpg","png","jpeg"])

if image_streamlit != None:
    st.image(image_streamlit)
    with open("image.jpg",mode = "wb") as f:
        f.write(image_streamlit.getbuffer())
    img = model.predict(source="image.jpg",
                        stream=True, retina_masks=True)
    new = None
    original_img = cv2.imread("image.jpg")
    for result in img:
        mask = result.masks.cpu().numpy()
        masks = mask.masks.astype(bool)
        
        ori_img = result.orig_img
        new = np.ones_like(ori_img, dtype=np.uint8)
#         new_color = np.array([0, 255, 0])  # Green color, you can adjust the values accordingly
#         for m in masks:
#             new[m] = ori_img[m]
        new_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color, you can adjust the values accordingly
        modified_img = original_img.copy()

        # Ensure the mask dimensions match the image dimensions
        if masks.shape[:2] == original_img.shape[:2]:
            modified_img[masks] = new_color
        else:
            st.error("Mask dimensions do not match the image dimensions.")

        # Save and display the modified image
        cv2.imwrite("modified_image.png", modified_img)
        st.image("modified_image.png")

