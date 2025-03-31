import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
from pages.face.main import Facedetect


# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_image_from_upload, 
                   get_facial_landmarks, 
                   draw_facial_landmarks, 
                   process_webcam_image)

# Set page configuration
st.set_page_config(
    page_title="Register Faces",
    page_icon="👁️",
    layout="wide"
)

face_detect = Facedetect()

# Page title
st.title("Register Faces")
st.write("Upload an image or use your webcam to detect facial landmarks and register your name.")

# Create tabs for image upload and webcam
tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])

with tab1:

    st.header("Enter person name")
    user_name_ = st.text_input("Enter your name:", "", key="text_input_upload")

    st.header("Upload an Image") 
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = load_image_from_upload(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="RGB", use_column_width=True)
        
        # Process image for landmarks
        with st.spinner("Detecting facial landmarks..."):
            face_landmarks_list, face_locations = get_facial_landmarks(image)
            
            if face_landmarks_list:
                # Draw landmarks on the image
                image_with_landmarks = draw_facial_landmarks(image, face_landmarks_list)
                
                # Display image with landmarks
                st.subheader("Facial Landmarks Detected")
                st.image(image_with_landmarks, channels="RGB", use_column_width=True)
                
                # Display information about detected features
                st.subheader("Detected Features")
                st.write(f"Found {len(face_landmarks_list)} face(s) in the image")
                
                # Expand to show landmark details
                with st.expander("View landmark details"):
                    for i, landmarks in enumerate(face_landmarks_list):
                        st.write(f"Face #{i+1}")
                        for feature, points in landmarks.items():
                            st.write(f"- {feature.replace('_', ' ').title()}: {len(points)} points")
            else:
                st.warning("No faces detected in the image. Please try another image.")

        image_path = f"data/{user_name_}.jpg"
        cv2.imwrite(image_path, image)
        feature_point = face_detect.feature(image)
        face_detect.write_to_csv(user_name_, image_path, feature_point)

with tab2:
    st.header("Enter person name")
    user_name = st.text_input("Enter your name:", "", key="text_input_webcam")


    st.header("Use Webcam")
    
    # Webcam input
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Read image from the buffer
        bytes_data = img_file_buffer.getvalue()
        
        # Convert to an OpenCV image
        file = tempfile.NamedTemporaryFile(delete=False)
        file.write(bytes_data)
        file.close()
        
        img = cv2.imread(file.name)

        os.unlink(file.name) 

        image_path = f"pages/face/data/images/{user_name}.jpg"
        cv2.imwrite(image_path, img)
        feature_point = face_detect.feature(img)
        face_detect.write_to_csv(user_name, image_path, feature_point)

        if feature_point:
            st.warning("Face registered successfully.")
        else:
            st.warning("No faces detected in the image. Please try again.")

# Footer with return to main page
st.markdown("---")
st.page_link("app.py", label="Return to Main Page", icon="🏠")
