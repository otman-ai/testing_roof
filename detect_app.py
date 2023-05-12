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
         #cv2.imwrite("mask.jpg",mask.masks)
         masks = mask.masks.astype(bool)
        
         ori_img = result.orig_img
         new = np.ones_like(ori_img, dtype=np.uint8)
         new_ = np.ones_like(ori_img, dtype=np.uint8)

         for m in masks:
             new[m] = ori_img[m]
             new_[m] = 0
         cv2.imwrite("modified_image.png", new)
         #cv2.imwrite("modified_image_.png", new_)
     
         #st.image("modified_image.png")
         st.image(mask.masks)
          
         mask = cv2.imread("modified_image.png")
         _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
         green_hair = np.copy(original_img)
         green_hair[(mask==255).all(-1)] = [0,255,0]
         st.image(green_hair)

# from ultralytics import YOLO
# import streamlit as st
# import cv2
# import numpy as np

# st.title("Wall Extraction and Color Change")
# st.write("Made by Otman Heddouch")
# st.write("Please upload an image of a house with a roof. The extracted wall will be displayed above after a few seconds.")
# st.info("Make sure the wall is clear in the image.")

# # Load the model
# @st.cache_resource
# def load_model():
#     model = YOLO("wall.pt")
#     return model

# model = load_model()

# image_streamlit = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

# if image_streamlit is not None:
#     st.image(image_streamlit)
#     with open("image.jpg", mode="wb") as f:
#         f.write(image_streamlit.getbuffer())
    
#     img = model.predict(source="image.jpg", stream=True, retina_masks=True)
#     original_img = cv2.imread("image.jpg")
    
#     for result in img:
#         mask = result.masks.cpu().numpy()
#         masks = mask.masks.astype(bool)
        
#         modified_img = original_img.copy()
#         new_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color, you can adjust the values accordingly
#         modified_img[masks] = new_color
        
#         cv2.imwrite("modified_image.png", modified_img)
#         st.image("modified_image.png")

