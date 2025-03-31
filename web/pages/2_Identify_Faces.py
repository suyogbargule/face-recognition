import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import time
from pages.face.main import Facedetect

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_image_from_upload, get_face_locations, draw_face_boxes, 
                   process_webcam_image)

# Set page configuration
st.set_page_config(
    page_title="Identify Faces",
    page_icon="🔍",
    layout="wide"
)

face_detect = Facedetect()
face_detect.load_csv_to_dict()

# Page title
st.title("Identify Faces")
st.write("Detect and identify faces using your webcam or uploaded images")

# Create a two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Detected Faces")
    st.write("Faces detected from your webcam or uploaded image will appear here.")
    
    # Placeholder for detected faces
    detected_faces_placeholder = st.empty()

with col2:
    st.header("Face Detection")
    
    # Create tabs for image upload and webcam
    identify_tab1, identify_tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    with identify_tab1:
        # File uploader for identification
        identify_file = st.file_uploader("Upload image for face detection", type=["jpg", "jpeg", "png"])
        
        if identify_file is not None:
            # Load the image
            identify_image = load_image_from_upload(identify_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(identify_image, channels="RGB", use_column_width=True)
            
            # Process image for face detection
            with st.spinner("Detecting faces..."):
                # Get face locations
                face_locations = get_face_locations(identify_image)
                
                if face_locations:
                    # Draw boxes around faces
                    image_with_boxes = draw_face_boxes(identify_image, face_locations)
                    
                    # Display image with boxes
                    st.subheader("Detection Results")
                    st.image(image_with_boxes, channels="RGB", use_column_width=True)
                    
                    # Update the detected faces placeholder
                    with detected_faces_placeholder.container():
                        st.subheader(f"Found {len(face_locations)} face(s)")
                        
                        # Create a grid to display detected faces
                        faces_cols = st.columns(min(3, len(face_locations)))
                        
                        for i, (top, right, bottom, left) in enumerate(face_locations):
                            # Extract face
                            face_img = identify_image[top:bottom, left:right]
                            
                            # Display each detected face
                            with faces_cols[i % 3]:
                                st.image(face_img, caption=f"Face #{i+1}", width=150)
                                st.write(f"Position: ({left}, {top}), Size: {right-left}x{bottom-top}")
                else:
                    st.warning("No faces detected in the image. Please try another image.")
                    detected_faces_placeholder.empty()
    
    with identify_tab2:
        st.subheader("Use Webcam")
        
        # Webcam input
        img_file_buffer = st.camera_input("Take a picture")
        
        if img_file_buffer is not None:
            # Read image from the buffer and convert to numpy array
            bytes_data = img_file_buffer.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            
            # Convert to an OpenCV image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Pass the OpenCV image to face_detect.detect()
            face_detect.detect(img)


# Footer with return to main page
st.markdown("---")
st.page_link("app.py", label="Return to Main Page", icon="🏠")
