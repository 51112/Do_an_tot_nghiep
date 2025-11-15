# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from pytube import YouTube
from PIL import Image
import time
import os
from collections import defaultdict, deque
from typing import List

st.set_page_config(page_title="Traffic Monitoring (Detection + Tracking + Counting)", layout="wide")
st.title("🚦 Traffic Monitoring — Detection · Tracking · Counting")

# ---------------------------
# Load model (local best.pt)
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        raise FileNotFoundError("Không tìm thấy best.pt trong repo! Hãy upload vào Hugging Face Space.")
    return YOLO("best.pt")

model = load_model()

# ---------------------------
# Simple tracker
# ---------------------------
class Track:
    def __init__(self, tid, box, label, score):
        self.id = tid
        self.box = box
        self.label = label
        self.score = score
        self.missed = 0
        self.history = deque(maxlen=40)

    def center(self):
        x1,y1,x2,y2 = self.box
        return np.array([(x1+x2)/2, (y1+y2)/2])

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=12):
        self.next_id = 1
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.seen_ids = defaultdict(set)

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = max(0,(a[2]-a[0])) * max(0,(a[3]-a[1]))
        areaB = max(0,(b[2]-b[0])) * max(0,(b[3]-b[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, dets):
        used = set()
        ids = list(self.tracks.keys())

        # Match with IoU
        for tid in ids:
            t = self.tracks[tid]
            best = -1
            best_j = -1
            for j, det in enumerate(dets):
                if j in used: continue
                iou = self.iou(t.box, det["box"])
                if iou > best:
                    best = iou
                    best_j = j

            if best >= self.iou_threshold:
                det = dets[best_j]
                t.box = det["box"]
                t.label = det["label"]
                t.score = det["score"]
                t.history.append(tuple(t.center()))
                t.missed = 0
                used.add(best_j)
                self.seen_ids[t.label].add(t.id)
            else:
                t.missed += 1

        # Register new
        for j, det in enumerate(dets):
            if j not in used:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, det["box"], det["label"], det["score"])
                tr.history.append(tuple(tr.center()))
                self.tracks[tid] = tr
                self.seen_ids[tr.label].add(tid)

        # Remove lost
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

    def counts(self):
        return {k: len(v) for k,v in self.seen_ids.items()}

# ---------------------------
# Class filter
# ---------------------------
DEFAULT_CLASS_MAP = {0:"person",1:"bicycle",2:"car",3:"motorbike",5:"bus",7:"truck"}
VEHICLES = {"car","motorbike","bus","truck","bicycle"}

def yolo_detect(frame_rgb, conf):
    res = model(frame_rgb, imgsz=640, conf=conf, verbose=False)[0]
    dets = []
    for box,score,cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
        label = DEFAULT_CLASS_MAP.get(cls, f"class{cls}")
        if label in VEHICLES:
            dets.append({"box": box, "label": label, "score": float(score)})
    return dets

def draw_tracks(frame, tracks):
    for t in tracks.values():
        x1,y1,x2,y2 = t.box.astype(int)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{t.label} ID:{t.id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    return frame

# ---------------------------
# Full pipeline
# ---------------------------
def process_video(path, conf=0.25, skip=2):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(3)); h = int(cap.get(4))

    out_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(out_temp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = SimpleTracker()
    frame_id = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        if frame_id % skip != 0:
            writer.write(frame)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = yolo_detect(rgb, conf)
        tracker.update(dets)

        frame = draw_tracks(frame, tracker.tracks)
        writer.write(frame)

        if total:
            bar.progress(frame_id / total)

    cap.release()
    writer.release()
    return out_temp.name, tracker.counts()

# ---------------------------
# UI
# ---------------------------
st.sidebar.subheader("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25)
skip = st.sidebar.slider("Skip frames", 1, 8, 2)

tab1, tab2 = st.tabs(["📤 Upload Video", "🔗 YouTube Link"])

with tab1:
    file = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        st.video(tmp.name)

        if st.button("Run detection + tracking"):
            out, counts = process_video(tmp.name, conf, skip)
            st.video(out)
            st.write("### Counts:", counts)

with tab2:
    url = st.text_input("YouTube Video URL")
    if url and st.button("Process YouTube Video"):
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4').order_by("resolution").first()
        yt_path = stream.download(filename="yt.mp4")
        out, counts = process_video(yt_path, conf, skip)
        st.video(out)
        st.write("### Counts:", counts)
