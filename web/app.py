import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Welcome to Facial Recognition App")
st.write("""
This application provides facial recognition features using your webcam or uploaded images.
Choose one of the options below to get started:
""")

# Create two columns for the buttons
col1, col2 = st.columns(2)

with col1:
    st.subheader("Register faces")
    st.write("Detect and visualize facial landmarks")
    st.page_link("pages/1_Extract_Feature_Points.py", label="Go to Register faces", icon="👁️")

with col2:
    st.subheader("Identify Faces")
    st.write("Recognize and identify faces in images or webcam")
    st.page_link("pages/2_Identify_Faces.py", label="Go to Face Identification", icon="🔍")

# Add information about the app
st.markdown("---")
st.subheader("How it works")
st.write("""
- **Facial Feature Points**: Detects key landmarks on faces including eyes, nose, mouth, and jawline
- **Face Identification**: Compares faces with known samples to identify individuals

This application uses computer vision and machine learning technologies to analyze facial features.
""")

# Footer
st.markdown("---")
st.caption("Facial Recognition App | Built with Streamlit, OpenCV and face_recognition")
