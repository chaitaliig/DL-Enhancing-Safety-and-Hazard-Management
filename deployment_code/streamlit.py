import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\panka\Desktop\deployment_code\weights\best.pt")

# Set page configuration
st.set_page_config(page_title="Hazardous Material Detection", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #ffffff, #d3e0ea);
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background: #222;
        color: white;
    }
    h1 {
        color: #007acc;
        text-align: center;
        font-size: 3rem;
    }
    h2 {
        color: #007acc;
    }
    .stButton>button {
        color: white;
        background-color: #007acc;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title with a logo
st.image(r"C:\Users\panka\Downloads\images.png", width=100)
st.title("Hazardous Material Detection")
st.markdown("### Upload a video or image to detect hazardous materials using YOLOv8")

# Sidebar customization
st.sidebar.image(r"C:\Users\panka\Downloads\images (5).jpg", width=150)
st.sidebar.title("Upload Options")
st.sidebar.markdown("### Choose whether to upload a video or an image.")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    upload_option = st.selectbox("What would you like to upload?", ("Video", "Image"))

with col2:
    if upload_option == "Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            # Create a temporary file to store the uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # Load the video with OpenCV
            video = cv2.VideoCapture(tfile.name)

            stframe = st.empty()  # Placeholder to display frames

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Perform detection on the frame
                results = model(frame)[0]  # Get the first result

                # Draw bounding boxes and labels on the frame
                for result in results.boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                    conf = result.conf[0]  # Confidence score
                    cls = int(result.cls[0])  # Class ID
                    label = f"{model.names[cls]} {conf:.2f}"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add a background rectangle for text
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_w, label_h = label_size
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), cv2.FILLED)

                    # Add the text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Display the frame
                stframe.image(frame, channels="BGR", use_column_width=True)

            video.release()

    elif upload_option == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Load the image with PIL
            image = Image.open(uploaded_image)
            image_np = np.array(image)

            # Perform detection on the image
            results = model(image_np)[0]  # Get the first result

            # Draw bounding boxes and labels on the image
            for result in results.boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                conf = result.conf[0]  # Confidence score
                cls = int(result.cls[0])  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add a background rectangle for text
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_w, label_h = label_size
                cv2.rectangle(image_np, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), cv2.FILLED)

                # Add the text
                cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the image
            st.image(image_np, channels="RGB", use_column_width=True)

st.markdown("<br><br>Upload a video or an image file to see hazardous material detection in action.", unsafe_allow_html=True)