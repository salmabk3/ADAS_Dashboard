import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="ADAS Dashboard (YOLO + MiDaS)", layout="wide")
st.title("🚗 ADAS Dashboard — YOLOv8 + MiDaS (Streamlit Cloud)")
st.caption("Cloud = vidéo uploadée. Pas de webcam/son. Dashboard = supervision + stats + graphes.")

# ===============================
# MODES D'ÉCLAIRAGE
# ===============================
def apply_day(frame):
    return frame

def apply_low_light(frame):
    alpha = 0.8
    beta = -40
    out = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    noise = np.random.normal(0, 8, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

def apply_night(frame):
    gamma = 1.8
    invGamma = 1.0 / gamma
    table = (np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8"))
    out = cv2.LUT(frame, table)
    out = cv2.convertScaleAbs(out, alpha=0.85, beta=-25)

    b, g, r = cv2.split(out)
    b = cv2.convertScaleAbs(b, alpha=1.10, beta=5)
    r = cv2.convertScaleAbs(r, alpha=0.90, beta=0)
    out = cv2.merge([b, g, r])

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b2 = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b2]), cv2.COLOR_LAB2BGR)

    noise = np.random.normal(0, 4, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

def pick_filter(mode):
    if mode == "day":
        return apply_day
    if mode == "low":
        return apply_low_light
    return apply_night

# ===============================
# CHARGEMENT MODÈLES (Cloud)
# ===============================
@st.cache_resource
def load_models():
    model_yolo = YOLO("yolov8n.pt")
    model_depth = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model_depth.to("cpu").eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return model_yolo, model_depth, transform

model_yolo, model_depth, transform = load_models()

# ===============================
# DETECTION
# ===============================
def detect_frame(frame_bgr, conf_thr=0.4, zone_y=300, red_thr=3.0, yellow_thr=6.0):
    frame_resized = cv2.resize(frame_bgr, (640, 480))

    # YOLO
    results = model_yolo(frame_resized, verbose=False, conf=float(conf_thr))

    # MiDaS
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to("cpu")
    with torch.no_grad():
        prediction = model_depth(input_batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_resized.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    danger_level = "green"
    pedestrian_count = 0
    min_distance = None

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cx = max(0, min(cx, 639))
            cy = max(0, min(cy, 479))

            depth_value = depth_norm[cy, cx]
            distance = 1.0 / (depth_value + 1e-6)

            pedestrian_count += 1
            min_distance = distance if min_distance is None else min(min_distance, distance)

            pedestrian_in_zone = y2 > int(zone_y)
            if pedestrian_in_zone and distance < float(red_thr):
                danger_level = "red"
            elif pedestrian_in_zone and distance < float(yellow_thr) and danger_level != "red":
                danger_level = "yellow"

            if danger_level == "red":
                color = (0, 0, 255)
            elif danger_level == "yellow":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, f"{distance:.1f} m",
                        (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.line(frame_resized, (0, int(zone_y)), (639, int(zone_y)), (255, 255, 255), 2)
    if danger_level == "red":
        cv2.putText(frame_resized, "⚠️ PEDESTRIAN DANGER",
                    (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return frame_resized, danger_level, pedestrian_count, min_distance

def state_to_num(s):
    return {"green": 0, "yellow": 1, "red": 2}.get(s, 0)

# ===============================
# SIDEBAR PARAMS
# ===============================
with st.sidebar:
    st.header("⚙️ Paramètres")

    mode = st.selectbox("Mode éclairage", ["day", "low", "night"], index=0)

    conf_day = st.slider("YOLO conf (day)", 0.05, 0.90, 0.40, 0.01)
    conf_low = st.slider("YOLO conf (low)", 0.05, 0.90, 0.30, 0.01)
    conf_night = st.slider("YOLO conf (night)", 0.05, 0.90, 0.25, 0.01)

    zone_y = st.slider("Zone proche (y2 > ...)", 100, 479, 300, 1)

    red_thr = st.slider("Seuil ROUGE (m)", 1.0, 15.0, 3.0, 0.1)
    yellow_thr = st.slider("Seuil JAUNE (m)", 1.0, 25.0, 6.0, 0.1)

    max_frames = st.slider("Frames à traiter (démo cloud)", 30, 900, 240, 10)
    run_btn = st.button("▶️ Lancer le dashboard")

video_file = st.file_uploader("📤 Upload une vidéo (.mp4)", type=["mp4", "mov", "m4v", "avi"])

colL, colR = st.columns([1.25, 1.0])
frame_slot = colL.empty()
kpi_slot = colR.empty()
chart1_slot = colR.empty()
chart2_slot = colR.empty()

# session stats
if "t_hist" not in st.session_state:
    st.session_state.t_hist = []
    st.session_state.min_d_hist = []
    st.session_state.state_hist = []
    st.session_state.yellow_frames = 0
    st.session_state.red_frames = 0

def pick_conf(mode):
    if mode == "day":
        return conf_day
    if mode == "low":
        return conf_low
    return conf_night

if run_btn:
    if video_file is None:
        st.error("❌ Upload une vidéo d’abord.")
        st.stop()

    # reset stats
    st.session_state.t_hist = []
    st.session_state.min_d_hist = []
    st.session_state.state_hist = []
    st.session_state.yellow_frames = 0
    st.session_state.red_frames = 0

    tmp_path = "uploaded_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Impossible de lire la vidéo.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    proc_filter = pick_filter(mode)
    conf_thr = float(pick_conf(mode))

    processed = 0
    while processed < int(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = proc_filter(frame)

        annotated, danger_level, ped_count, min_distance = detect_frame(
            frame,
            conf_thr=conf_thr,
            zone_y=int(zone_y),
            red_thr=float(red_thr),
            yellow_thr=float(yellow_thr),
        )

        # counters
        if danger_level == "yellow":
            st.session_state.yellow_frames += 1
        elif danger_level == "red":
            st.session_state.red_frames += 1

        # history
        t = processed / float(fps)
        st.session_state.t_hist.append(float(t))
        st.session_state.min_d_hist.append(float(min_distance) if min_distance is not None else np.nan)
        st.session_state.state_hist.append(danger_level)

        # render frame
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

        # KPIs
        state_txt = "SAFE" if danger_level == "green" else ("WARNING" if danger_level == "yellow" else "DANGER")
        kpi_slot.markdown(
            f"### État ADAS : **{state_txt}**\n"
            f"- Mode : **{mode}**\n"
            f"- YOLO conf : **{conf_thr:.2f}**\n"
            f"- Piétons : **{ped_count}**\n"
            f"- Distance min : **{(min_distance if min_distance is not None else float('nan')):.2f} m**\n\n"
            f"**Frames JAUNES** : {st.session_state.yellow_frames}  \n"
            f"**Frames ROUGES** : {st.session_state.red_frames}"
        )

        # ✅ CHARTS (corrigé avec DataFrame)
        N = min(250, len(st.session_state.t_hist))
        t_hist = st.session_state.t_hist[-N:]
        d_hist = st.session_state.min_d_hist[-N:]
        s_hist = [state_to_num(x) for x in st.session_state.state_hist[-N:]]

        N2 = min(len(t_hist), len(d_hist), len(s_hist))
        df_chart = pd.DataFrame({
            "time_s": t_hist[:N2],
            "min_distance_m": d_hist[:N2],
            "state": s_hist[:N2],
        }).set_index("time_s")

        chart1_slot.line_chart(df_chart[["min_distance_m"]])
        chart2_slot.line_chart(df_chart[["state"]])

        processed += 1

    cap.release()
    st.success(f"✅ Terminé : {processed} frames traitées (mode={mode}).")
