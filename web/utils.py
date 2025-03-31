import cv2
import numpy as np
from PIL import Image
import streamlit as st

def load_image_from_upload(uploaded_file):
    """Load an image from a file upload widget"""
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Convert to RGB (face_recognition uses RGB)
        image = image.convert('RGB')
        # Convert to numpy array
        image_array = np.array(image)
        return image_array
    return None

def get_face_locations(image):
    """Get face locations from an image"""
    #if image is not None:
        # Locate faces in the image
        #face_locations = face_recognition.face_locations(image)
        #return face_locations
    return []

def get_facial_landmarks(image):
    """Get facial landmarks from an image"""
    # if image is not None:
    #     # Get face locations
    #     face_locations = face_recognition.face_locations(image)
    #     # Get facial landmarks
    #     #face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
    #     return face_landmarks_list, face_locations
    return [], []

def draw_facial_landmarks(image, face_landmarks_list):
    """Draw facial landmarks on an image"""
    if image is None or not face_landmarks_list:
        return image
    
    # Create a copy of the image to draw on
    image_with_landmarks = image.copy()
    
    # Define colors for different parts of the face
    colors = {
        'chin': (0, 255, 0),        # Green
        'left_eyebrow': (255, 0, 0), # Blue
        'right_eyebrow': (255, 0, 0), # Blue
        'nose_bridge': (0, 0, 255),  # Red
        'nose_tip': (0, 0, 255),     # Red
        'left_eye': (255, 255, 0),   # Cyan
        'right_eye': (255, 255, 0),  # Cyan
        'top_lip': (255, 0, 255),    # Magenta
        'bottom_lip': (255, 0, 255)  # Magenta
    }
    
    # Draw the landmarks for each face
    for face_landmarks in face_landmarks_list:
        # Draw each facial feature
        for facial_feature, landmark_points in face_landmarks.items():
            # Draw lines connecting the points
            color = colors.get(facial_feature, (0, 255, 0))  # Default to green
            points = np.array(landmark_points, dtype=np.int32)
            
            # Draw circles at each point
            for point in points:
                cv2.circle(image_with_landmarks, tuple(point), 2, color, -1)
            
            # Connect the points with lines
            if len(points) > 1:
                cv2.polylines(image_with_landmarks, [points], 
                              facial_feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip'], 
                              color, 2)
    
    return image_with_landmarks

def draw_face_boxes(image, face_locations):
    """Draw boxes around detected faces"""
    if image is None or not face_locations:
        return image
    
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    
    # Draw a rectangle around each face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
    
    return image_with_boxes

def generate_face_encodings(image, face_locations):
    """Generate face encodings for recognition"""
    # if image is not None and face_locations:
    #     face_encodings = face_recognition.face_encodings(image, face_locations)
    #     return face_encodings
    return []

def compare_faces(known_face_encodings, face_encoding, tolerance=0.6):
    """Compare a face encoding with a list of known face encodings"""
    matches = 0
    # if not known_face_encodings or face_encoding is None:
    #     return []
    
    # matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)
    return matches

def get_face_distance(known_face_encodings, face_encoding):
    """Get face distances between encodings"""
    face_distances =0
    # if not known_face_encodings or face_encoding is None:
    #     return []
    
    # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return face_distances

def process_webcam_image(img):
    """Process webcam image for face_recognition library"""
    # Convert RGB to BGR (opencv uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert back to RGB (face_recognition uses RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
