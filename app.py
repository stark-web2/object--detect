remove only this part to the code import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Turn on your webcam to detect and track objects in real time.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# VIDEO PROCESSOR CLASS
# -----------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO tracking (this gives object IDs)
        results = self.model.track(img, persist=True, verbose=False)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -----------------------------
# START WEBCAM STREAM
# -----------------------------
webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

# -----------------------------
# INFO SECTION (FOR REPORT REQUIREMENTS)
# -----------------------------
st.markdown("""
---

## 📊 Expected Outputs

### 1. Functional Web App
- Live webcam feed
- YOLO object detection overlay

### 2. Live Detection
- Bounding boxes appear instantly
- Labels like person, phone, bottle, etc.

### 3. Object Tracking
- Objects keep IDs while moving
- Smooth real-time tracking using YOLOv8

---

## 📌 For Your Report

### Observations
- List detected objects
- Lighting effect on accuracy
- Performance (lag/smooth)

### Screenshots
Capture:
- People detection
- Multiple object detection
- Moving object tracking

### Reflection
- What objects were easiest to detect?
- What affected accuracy?

---

## 🚀 Optional Enhancements
- Object counting (people, cars)
- Alerts for specific objects
- Save detected frames as images
""")
