# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import yt_dlp
import time
import os
import threading
from collections import defaultdict, deque

st.set_page_config(page_title="Traffic Live Monitoring", layout="wide")
st.title("Traffic Monitoring — Live Detection & Counting")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("Không tìm thấy `best.pt`! Upload vào root repo.")
        return None
    return YOLO("best.pt")

model = load_model()
if model is None:
    st.stop()

# ---------------------------
# Tracker
# ---------------------------
class Track:
    def __init__(self, tid, box, label, score):
        self.id = tid; self.box = box; self.label = label; self.score = score
        self.missed = 0; self.history = deque(maxlen=40)
    def center(self):
        x1,y1,x2,y2 = self.box
        return np.array([(x1+x2)/2, (y1+y2)/2])

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=12):
        self.next_id = 1; self.tracks = {}; self.iou_threshold = iou_threshold
        self.max_missed = max_missed; self.seen_ids = defaultdict(set)

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = max(0,(a[2]-a[0])) * max(0,(a[3]-a[1]))
        areaB = max(0,(b[2]-b[0])) * max(0,(b[3]-b[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, dets):
        used = set(); ids = list(self.tracks.keys())
        for tid in ids:
            t = self.tracks[tid]; best = -1; best_j = -1
            for j, det in enumerate(dets):
                if j in used: continue
                iou_val = self.iou(t.box, det["box"])
                if iou_val > best:
                    best = iou_val; best_j = j
            if best >= self.iou_threshold:
                det = dets[best_j]
                t.box = det["box"]; t.label = det["label"]; t.score = det["score"]
                t.history.append(tuple(t.center())); t.missed = 0
                used.add(best_j); self.seen_ids[t.label].add(t.id)
            else:
                t.missed += 1
        for j, det in enumerate(dets):
            if j not in used:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, det["box"], det["label"], det["score"])
                tr.history.append(tuple(tr.center()))
                self.tracks[tid] = tr; self.seen_ids[tr.label].add(tid)
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
    def counts(self):
        return {k: len(v) for k,v in self.seen_ids.items()}

# ---------------------------
# Detection & Draw
# ---------------------------
VEHICLES = {"car","motorbike","bus","truck","bicycle"}
DEFAULT_CLASS_MAP = {0:"person",1:"bicycle",2:"car",3:"motorbike",5:"bus",7:"truck"}

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
# Real-time Processor (Upload)
# ---------------------------
def process_upload_realtime(path, conf, skip, ph_video, ph_count):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        ph_video.error("Không mở được video!")
        return
    tracker = SimpleTracker(); frame_id = 0; start = time.time(); proc = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % skip != 0: continue
        frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = yolo_detect(rgb, conf); tracker.update(dets); frame = draw_tracks(frame, tracker.tracks)
        proc += 1
        if proc % 30 == 0:
            fps = proc / (time.time() - start + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        ph_video.image(buf.tobytes(), channels="BGR", use_column_width=True)
        ph_count.write(f"**Đếm hiện tại**: {tracker.counts()}")
    cap.release()
    ph_count.success(f"**Tổng đếm**: {tracker.counts()}")

# ---------------------------
# YouTube Live Processor (CHỈ DÙNG yt-dlp + HLS)
# ---------------------------
stop_event = threading.Event()

def youtube_live_processor(video_id, conf, skip, ph_video, ph_count):
    yt_path = "live_stream.mp4"
    try:
        ph_count.info("Đang kết nối YouTube Live...")

        ydl_opts = {
            'format': 'worst[ext=mp4]',  # Dùng worst để load nhanh, tránh lỗi
            'outtmpl': yt_path,
            'quiet': True,
            'no_warnings': True,
            'continuedl': True,
            'wait_for_video': (10, 30),  # Chờ 10-30s nếu chưa có stream
            'live_from_start': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if not info.get('is_live'):
                ph_video.error("Video này KHÔNG PHẢI live stream! Vui lòng dùng video đang phát trực tiếp.")
                return
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        if not os.path.exists(yt_path):
            ph_video.error("Không tải được stream!")
            return

        cap = cv2.VideoCapture(yt_path)
        if not cap.isOpened():
            ph_video.error("Không mở được file stream!")
            return

        tracker = SimpleTracker(); frame_id = 0; start = time.time(); proc = 0
        ph_count.success("Kết nối thành công! Đang xử lý live...")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            frame_id += 1
            if frame_id Historical % skip != 0: continue
            frame = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = yolo_detect(rgb, conf); tracker.update(dets); frame = draw_tracks(frame, tracker.tracks)
            proc += 1
            if proc % 30 == 0:
                fps = proc / (time.time() - start + 1e-6)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            ph_video.image(buf.tobytes(), channels="BGR", use_column_width=True)
            ph_count.write(f"**YouTube Live - Đếm hiện tại**: {tracker.counts()}\n**Frame**: {frame_id}")

        cap.release()
        ph_count.success(f"**Tổng đếm**: {tracker.counts()}")

    except Exception as e:
        ph_video.error(f"Lỗi: {str(e)}")
    finally:
        if os.path.exists(yt_path): os.remove(yt_path)
        stop_event.clear()

# ---------------------------
# UI
# ---------------------------
st.sidebar.subheader("Cài đặt")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25)
skip = st.sidebar.slider("Skip frames", 1, 5, 2)

tab1, tab2 = st.tabs(["Upload Video", "YouTube Live"])

# --- TAB 1: Upload ---
with tab1:
    file = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read()); tmp.close()
        st.video(tmp.name)
        if st.button("Run Detection"):
            ph_v = st.empty(); ph_c = st.empty()
            process_upload_realtime(tmp.name, conf, skip, ph_v, ph_c)
            os.unlink(tmp.name)

# --- TAB 2: YouTube Live ---
with tab2:
    st.subheader("YouTube Live Stream")
    st.info("Chỉ dùng với **video đang LIVE**. Ví dụ: `xCNRP131kNY` (nếu đang phát).")

    live_id = st.text_input("YouTube Live ID", placeholder="xCNRP131kNY")
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("Bắt đầu Live", type="primary")
    with col2:
        stop_btn = st.button("Dừng", type="secondary")

    ph_video = st.empty()
    ph_count = st.empty()

    if start_btn:
        if not live_id.strip():
            st.error("Nhập ID!")
            st.stop()
        stop_event.clear()
        thread = threading.Thread(
            target=youtube_live_processor,
            args=(live_id, conf, skip, ph_video, ph_count),
            daemon=True
        )
        thread.start()
        st.success("Đang kết nối...")

    if stop_btn:
        stop_event.set()
        st.warning("Đã dừng.")
