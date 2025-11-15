# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from pytubefix import YouTube
import yt_dlp
from PIL import Image
import time
import os
import threading
from collections import defaultdict, deque

st.set_page_config(page_title="Traffic Monitoring - Live & VOD", layout="wide")
st.title("Traffic Monitoring — Detection · Tracking · Counting")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Không tìm thấy file mô hình `best.pt`! Upload vào root repo.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Lỗi load mô hình: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# ---------------------------
# Tracker
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
        for tid in ids:
            t = self.tracks[tid]
            best = -1; best_j = -1
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
        for j, det in enumerate(dets):
            if j not in used:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, det["box"], det["label"], det["score"])
                tr.history.append(tuple(tr.center()))
                self.tracks[tid] = tr
                self.seen_ids[tr.label].add(tid)
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

    def counts(self):
        return {k: len(v) for k,v in self.seen_ids.items()}

# ---------------------------
# Detection & Draw
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
# Real-time Processor (Upload & VOD)
# ---------------------------
def process_video_realtime(path, conf=0.25, skip=2, placeholder_video=None, placeholder_count=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        if placeholder_video: placeholder_video.error("Không mở được video!")
        return None

    tracker = SimpleTracker()
    frame_id = 0
    fps_start = time.time()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % skip != 0: continue

        frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = yolo_detect(rgb, conf)
        tracker.update(dets)
        frame = draw_tracks(frame, tracker.tracks)

        processed += 1
        if processed % 30 == 0:
            fps = processed / (time.time() - fps_start + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if placeholder_video:
            placeholder_video.image(buffer.tobytes(), channels="BGR", use_column_width=True)
        if placeholder_count:
            placeholder_count.write(f"### Đang xử lý...\n**Đếm hiện tại**: {tracker.counts()}")

    cap.release()
    return tracker.counts()

# ---------------------------
# YouTube Live Stream Processor
# ---------------------------
stop_event = threading.Event()

def youtube_live_processor(video_id, conf=0.25, skip=2, placeholder_video=None, placeholder_count=None):
    yt_path = None
    try:
        placeholder_count.info("Đang tải YouTube Live stream...")
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': 'live_yt.%(ext)s',
            'quiet': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        yt_path = 'live_yt.mp4'
        if not os.path.exists(yt_path):
            raise Exception("Không tải được video Live!")

        cap = cv2.VideoCapture(yt_path)
        if not cap.isOpened():
            placeholder_video.error("Không mở được video Live!")
            return

        tracker = SimpleTracker()
        frame_id = 0
        fps_start = time.time()
        processed = 0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % skip != 0: continue

            frame = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = yolo_detect(rgb, conf)
            tracker.update(dets)
            frame = draw_tracks(frame, tracker.tracks)

            processed += 1
            if processed % 30 == 0:
                fps = processed / (time.time() - fps_start + 1e-6)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if placeholder_video:
                placeholder_video.image(buffer.tobytes(), channels="BGR", use_column_width=True)
            if placeholder_count:
                placeholder_count.write(f"### YouTube Live\n**Đếm hiện tại**: {tracker.counts()}\n**Frame**: {frame_id}")

        cap.release()
        final = tracker.counts()
        if placeholder_count:
            placeholder_count.success(f"### Hoàn thành!\n**Tổng đếm**: {final}")

    except Exception as e:
        if placeholder_video:
            placeholder_video.error(f"Lỗi: {str(e)}")
    finally:
        if yt_path and os.path.exists(yt_path):
            os.remove(yt_path)
        stop_event.clear()

# ---------------------------
# UI
# ---------------------------
st.sidebar.subheader("Cài đặt")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25)
skip = st.sidebar.slider("Skip frames", 1, 5, 2)

tab1, tab2, tab3 = st.tabs(["Upload Video", "YouTube Video", "YouTube Live"])

# --- TAB 1: Upload ---
with tab1:
    file = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read())
        tmp.close()
        st.video(tmp.name)

        if st.button("Run Detection"):
            video_ph = st.empty()
            count_ph = st.empty()
            counts = process_video_realtime(tmp.name, conf, skip, video_ph, count_ph)
            st.success("Hoàn thành!")
            st.write("### Tổng đếm:", counts)
            os.unlink(tmp.name)

# --- TAB 2: YouTube VOD ---
with tab2:
    url = st.text_input("YouTube Video URL (VOD)")
    if url and st.button("Process YouTube Video"):
        try:
            with st.spinner("Đang tải video..."):
                yt = YouTube(url, use_po_token=True)
                stream = yt.streams.filter(file_extension='mp4').order_by("resolution").first()
                if not stream:
                    st.error("Không tìm thấy stream MP4!")
                    st.stop()
                yt_path = stream.download(filename="yt.mp4")

            video_ph = st.empty()
            count_ph = st.empty()
            counts = process_video_realtime(yt_path, conf, skip, video_ph, count_ph)
            st.success("Hoàn thành!")
            st.write("### Tổng đếm:", counts)
            if os.path.exists(yt_path): os.remove(yt_path)
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- TAB 3: YouTube Live ---
with tab3:
    st.subheader("YouTube Live Stream Detection")
    st.info("Nhập **ID video Live** (phần sau `v=`). Ví dụ: `xCNRP131kNY`")

    live_id = st.text_input("YouTube Live ID", placeholder="Ví dụ: xCNRP131kNY")
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("Bắt đầu Live", type="primary")
    with col2:
        stop_btn = st.button("Dừng", type="secondary")

    video_ph = st.empty()
    count_ph = st.empty()

    if start_btn:
        if not live_id.strip():
            st.error("Vui lòng nhập ID!")
            st.stop()
        stop_event.clear()
        thread = threading.Thread(
            target=youtube_live_processor,
            args=(live_id, conf, skip, video_ph, count_ph),
            daemon=True
        )
        thread.start()
        st.success("Đã bắt đầu YouTube Live!")

    if stop_btn:
        stop_event.set()
        st.warning("Đã dừng.")
