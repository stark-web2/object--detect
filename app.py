import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.title("🎥 Live Object Detection & Tracking")
st.write("Turn on your webcam to detect, count objects, and trigger alerts in real time.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# ALERT SETTINGS
# -----------------------------
ALERT_CLASSES = ["knife", "cell phone", "scissors", "fire", "gun"]  # you can customize
PERSON_CLASS_ID = 0  # COCO dataset: person = 0

# Global stats (simple shared state)
st.session_state.setdefault("person_count", 0)
st.session_state.setdefault("alert_triggered", False)

# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(img, persist=True, verbose=False)[0]

        annotated_frame = img.copy()

        person_count = 0
        detected_classes = {}

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]

                # Count objects
                detected_classes[label] = detected_classes.get(label, 0) + 1

                # Count persons
                if cls_id == PERSON_CLASS_ID:
                    person_count += 1

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)

                # Alert color change
                if label in ALERT_CLASSES:
                    color = (0, 0, 255)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # -----------------------------
        # ALERT LOGIC
        # -----------------------------
        alert = any(obj in ALERT_CLASSES for obj in detected_classes)

        if alert:
            cv2.putText(
                annotated_frame,
                "⚠ ALERT OBJECT DETECTED!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # -----------------------------
        # DISPLAY COUNTS ON FRAME
        # -----------------------------
        cv2.putText(
            annotated_frame,
            f"People Count: {person_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        y_offset = 140
        for obj, count in detected_classes.items():
            cv2.putText(
                annotated_frame,
                f"{obj}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )
            y_offset += 25

        # Update session state (for UI display)
        st.session_state.person_count = person_count
        st.session_state.alert_triggered = alert

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
# SIDEBAR STATS
# -----------------------------
st.sidebar.header("📊 Live Stats")

st.sidebar.metric("👥 People Count", st.session_state.person_count)

if st.session_state.alert_triggered:
    st.sidebar.error("🚨 ALERT: Dangerous object detected!")
else:
    st.sidebar.success("No alerts detected")
