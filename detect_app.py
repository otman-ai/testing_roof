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
        #dummy_image = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
        #dummy_image[masks] = [255, 255, 255]
        #imsave('mask.png', masks)
        ori_img = result.orig_img
        new = np.ones_like(ori_img, dtype=np.uint8)
        #siding_mask = cv2.imread('mask.png', 0)  # Ensure the mask is grayscale (single channel)
        #siding_mask_3ch = cv2.cvtColor(siding_mask, cv2.COLOR_GRAY2BGR)
        new_color = np.array([0, 255, 0])  # Green color, you can adjust the values accordingly
        #modified_image = cv2.addWeighted(original_img, 1.0, siding_mask_3ch, 0.5, 0)
        #modified_image = cv2.addWeighted(modified_image, 1.0, new_color, 0.5, 0)
        for m in masks:
            new[m] = ori_img[m]

        cv2.imwrite("h.png",new)
        siding_mask_3ch = cv2.cvtColor(cv2.imread("h.png",0), cv2.COLOR_GRAY2BGR)
        modified_image = cv2.addWeighted(ori_img, 1.0, siding_mask_3ch, 0.5, 0)
        modified_image = cv2.addWeighted(modified_image, 1.0, new_color, 0.5, 0)
        st.image(modified_image)

