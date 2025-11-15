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
import queue
from collections import defaultdict, deque

# TẮT CẢNH BÁO
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["ULTRALYTICS_SETTINGS_DISABLE"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

# === CẤU HÌNH TRANG ===
st.set_page_config(page_title="Traffic Live Monitoring", layout="wide")
st.title("Traffic Monitoring — Live Detection (15s/lần)")

# === 1. ĐỌC LABELMAP.TXT ===
def load_label_map():
    if not os.path.exists("labelmap.txt"):
        st.error("Không tìm thấy `labelmap.txt`! Upload vào root repo.")
        return {}, {}
    id_to_label, label_to_id = {}, {}
    with open("labelmap.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line: continue
            parts = line.split(" ", 1)
            if len(parts) != 2: continue
            try:
                cid = int(parts[0])
                label = parts[1].strip()
                id_to_label[cid] = label
                label_to_id[label.lower()] = cid
            except: continue
    return id_to_label, label_to_id

ID_TO_LABEL, LABEL_TO_ID = load_label_map()
if not ID_TO_LABEL:
    st.stop()

# Chỉ detect xe
VEHICLE_LABELS = {"car", "motorbike", "bus", "truck"}
VEHICLE_IDS = {LABEL_TO_ID[label] for label in VEHICLE_LABELS if label in LABEL_TO_ID}

# === 2. TẢI MODEL ===
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("Không tìm thấy `best.pt`! Upload vào root repo.")
        return None
    return YOLO("best.pt")

model = load_model()
if model is None:
    st.stop()

# === 3. TRACKING ===
class Track:
    def __init__(self, tid, box, label, score):
        self.id = tid; self.box = box; self.label = label; self.score = score
        self.missed = 0; self.history = deque(maxlen=40)
    def center(self):
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=12):
        self.next_id = 1; self.tracks = {}; self.iou_threshold = iou_threshold
        self.max_missed = max_missed; self.seen_ids = defaultdict(set)

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0

    def update(self, dets):
        used = set(); ids = list(self.tracks.keys())
        for tid in ids:
            t = self.tracks[tid]; best_iou = -1; best_j = -1
            for j, det in enumerate(dets):
                if j in used: continue
                iou_val = self.iou(t.box, det["box"])
                if iou_val > best_iou:
                    best_iou = iou_val; best_j = j
            if best_iou >= self.iou_threshold:
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
                tr.history.append(tuple(tr.center())); self.tracks[tid] = tr
                self.seen_ids[tr.label].add(tid)
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
    def counts(self):
        return {k: len(v) for k, v in self.seen_ids.items()}

# === 4. DETECTION & DRAW ===
def yolo_detect(frame_rgb, conf):
    if not VEHICLE_IDS: return []
    results = model(frame_rgb, imgsz=640, conf=conf, verbose=False)[0]
    dets = []
    for box, score, cls in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy().astype(int)
    ):
        if cls not in VEHICLE_IDS: continue
        label = ID_TO_LABEL.get(cls, f"class{cls}")
        dets.append({"box": box, "label": label, "score": float(score)})
    return dets

def draw_tracks(frame, tracks):
    for t in tracks.values():
        x1, y1, x2, y2 = t.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{t.label} ID:{t.id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame

# === 5. XỬ LÝ 15s VIDEO (TỰ ĐỘNG LẶP) ===
stop_event = threading.Event()
message_queue = queue.Queue()
current_temp_file = None

def clear_all():
    global current_temp_file
    stop_event.set()
    if current_temp_file and os.path.exists(current_temp_file):
        try: os.unlink(current_temp_file)
        except: pass
    current_temp_file = None

def process_15s_segment(video_path, conf, skip, ph_video, ph_count, ph_live_video):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        message_queue.put(("status", "error", "Không mở được video!"))
        return

    tracker = SimpleTracker()
    frame_id = 0
    start_time = time.time()
    processed = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time > 15):  # Dừng sau 15s
            break
        frame_id += 1
        if frame_id % skip != 0: continue

        frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = yolo_detect(rgb, conf)
        tracker.update(dets)
        frame = draw_tracks(frame, tracker.tracks)

        processed += 1
        if processed % 10 == 0:
            fps = processed / (time.time() - start_time + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        message_queue.put(("frame", buffer.tobytes()))
        message_queue.put(("count", f"**Đếm hiện tại**: {tracker.counts()}"))

    cap.release()
    return tracker.counts()

def youtube_live_cycle(video_id, conf, skip):
    global current_temp_file
    total_time = 0
    max_duration = 180  # 3 phút = 180s
    segment_duration = 15  # 15s mỗi lần

    while total_time < max_duration and not stop_event.is_set():
        segment_start = time.time()
        yt_path = "segment.mp4"

        # TẢI 15s VIDEO
        try:
            ydl_opts = {
                'format': 'worst[ext=mp4]',
                'outtmpl': yt_path,
                'quiet': True, 'no_warnings': True,
                'download_ranges': lambda info, _: [{'start_time': 0, 'end_time': segment_duration}],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                if not info or not info.get('is_live'):
                    message_queue.put(("status", "error", "Video không phải live!"))
                    return
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except:
            message_queue.put(("status", "error", "Tải đoạn video thất bại!"))
            time.sleep(3)
            continue

        if not os.path.exists(yt_path):
            time.sleep(3)
            continue

        # Tạo file tạm để hiển thị
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.close()
        os.replace(yt_path, tfile.name)
        current_temp_file = tfile.name
        message_queue.put(("video", tfile.name))
        message_queue.put(("status", "info", f"Đang xử lý đoạn {int(total_time)}s - {int(total_time + 15)}s..."))

        # XỬ LÝ 15s
        ph_v = st.empty()
        ph_c = st.empty()
        final_count = process_15s_segment(tfile.name, conf, skip, ph_v, ph_c, None)

        # CLEAN
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)
        current_temp_file = None
        message_queue.put(("status", "success", f"Hoàn thành đoạn {int(total_time + 15)}s → Đếm: {final_count}"))

        total_time += (time.time() - segment_start)
        time.sleep(1)  # Nghỉ 1s trước đoạn tiếp theo

    message_queue.put(("status", "final", f"HOÀN TẤT! Tổng thời gian: {int(total_time)}s"))

# === 6. GIAO DIỆN ===
st.sidebar.header("Cài đặt")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
skip = st.sidebar.slider("Skip frames", 1, 5, 2)

tab1, tab2 = st.tabs(["Upload Video", "YouTube Live (15s/lần)"])

# --- TAB 1: Upload Video ---
with tab1:
    st.subheader("Upload Video để Test")
    uploaded_file = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read()); tfile.close()
        st.video(tfile.name)

        if st.button("Bắt đầu Phát Hiện", type="primary"):
            ph_v = st.empty(); ph_c = st.empty()
            with st.spinner("Đang xử lý..."):
                process_15s_segment(tfile.name, conf, skip, ph_v, ph_c, None)
            os.unlink(tfile.name)
            st.success("Hoàn tất!")

# --- TAB 2: YouTube Live (TỰ ĐỘNG 15s) ---
with tab2:
    st.subheader("YouTube Live — Tự động 15s/lần")
    st.info("**Tự động chạy 15s → clean → lặp → dừng sau 3 phút**")

    live_id = st.text_input("YouTube Live ID", placeholder="Ví dụ: xCNRP131kNY")
    start_btn = st.button("Bắt đầu Live (3 phút)", type="primary")

    ph_live_video = st.empty()
    ph_video = st.empty()
    ph_count = st.empty()

    # XỬ LÝ QUEUE
    try:
        while True:
            msg = message_queue.get_nowait()
            msg_type = msg[0]
            if msg_type == "video":
                ph_live_video.video(msg[1])
            elif msg_type == "frame":
                ph_video.image(msg[1], channels="BGR", use_container_width=True)
            elif msg_type == "count":
                ph_count.markdown(msg[1])
            elif msg_type == "status":
                status_type, text = msg[1], msg[2]
                if status_type == "info": ph_count.info(text)
                elif status_type == "success": ph_count.success(text)
                elif status_type == "error": ph_count.error(text)
                elif status_type == "final":
                    ph_count.success(text)
                    ph_video.empty(); ph_live_video.empty()
    except queue.Empty:
        pass

    if 'live_thread' not in st.session_state:
        st.session_state.live_thread = None

    if start_btn:
        id_val = live_id.strip()
        if not id_val:
            st.error("Nhập ID video!"); st.stop()
        if st.session_state.live_thread and st.session_state.live_thread.is_alive():
            st.warning("Đang chạy!"); st.stop()

        clear_all()
        stop_event.clear()
        st.session_state.live_thread = threading.Thread(
            target=youtube_live_cycle,
            args=(id_val, conf, skip),
            daemon=True
        )
        st.session_state.live_thread.start()
        st.success("Bắt đầu tự động 15s/lần...")
