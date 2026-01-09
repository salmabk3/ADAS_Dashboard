import time
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(page_title="ADAS Dashboard", layout="wide")
st.title("🚗 ADAS — Pedestrian Detection Dashboard (YOLO + MiDaS)")

# ---------------------------
# Filtres éclairage (comme votre notebook)
# ---------------------------
def apply_day(frame):
    return frame

def apply_low_light(frame):
    out = cv2.convertScaleAbs(frame, alpha=0.8, beta=-40)
    noise = np.random.normal(0, 8, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

def apply_night(frame):
    # nuit plus sombre mais encore exploitable
    out = cv2.convertScaleAbs(frame, alpha=0.6, beta=-70)
    b, g, r = cv2.split(out)
    b = cv2.convertScaleAbs(b, alpha=1.2, beta=5)
    g = cv2.convertScaleAbs(g, alpha=0.9, beta=0)
    r = cv2.convertScaleAbs(r, alpha=0.8, beta=0)
    out = cv2.merge([b, g, r])
    noise = np.random.normal(0, 6, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

# ---------------------------
# Chargement modèles (Cloud-friendly)
# ---------------------------
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")  # tu peux changer en yolov8s.pt si tu veux
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    device = "cpu"
    midas.to(device)
    return yolo, midas, transform, device

model_yolo, model_depth, transform, DEVICE = load_models()

# ---------------------------
# Sidebar paramètres (comme votre logique)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Paramètres ADAS")

    mode = st.selectbox("Mode éclairage", ["day", "low", "night"], index=0)

    conf_day = st.slider("YOLO conf (day)", 0.05, 0.90, 0.40, 0.01)
    conf_low = st.slider("YOLO conf (low)", 0.05, 0.90, 0.30, 0.01)
    conf_night = st.slider("YOLO conf (night)", 0.05, 0.90, 0.25, 0.01)

    zone_y = st.slider("Zone proche (y2 > ...)", 100, 479, 300, 1)

    red_day = st.slider("Seuil ROUGE (day) m", 1.0, 15.0, 4.5, 0.1)
    yellow_day = st.slider("Seuil JAUNE (day) m", 1.0, 25.0, 7.0, 0.1)

    red_night = st.slider("Seuil ROUGE (night) m", 1.0, 20.0, 5.5, 0.1)
    yellow_night = st.slider("Seuil JAUNE (night) m", 1.0, 30.0, 9.0, 0.1)

    max_frames = st.slider("Frames à traiter (démo cloud)", 50, 600, 200, 10)
    run_btn = st.button("▶️ Lancer analyse")

st.info("📌 Streamlit Cloud : upload une vidéo (mp4). La webcam/son ne marche pas sur Cloud.")

video_file = st.file_uploader("📤 Upload vidéo (.mp4)", type=["mp4", "mov", "m4v", "avi"])

colL, colR = st.columns([1.3, 1.0])
frame_box = colL.empty()
kpi_box = colR.empty()
chart1 = colR.empty()
chart2 = colR.empty()

# ---------------------------
# Session state (stats)
# ---------------------------
if "hist_t" not in st.session_state:
    st.session_state.hist_t = []
    st.session_state.hist_min_d = []
    st.session_state.hist_state = []
    st.session_state.yellow_frames = 0
    st.session_state.red_frames = 0

def state_to_num(s):
    return {"green": 0, "yellow": 1, "red": 2}.get(s, 0)

def pick_params():
    if mode == "day":
        return apply_day, conf_day, red_day, yellow_day
    if mode == "low":
        return apply_low_light, conf_low, red_day, yellow_day
    return apply_night, conf_night, red_night, yellow_night

if run_btn:
    st.session_state.hist_t = []
    st.session_state.hist_min_d = []
    st.session_state.hist_state = []
    st.session_state.yellow_frames = 0
    st.session_state.red_frames = 0

    if video_file is None:
        st.error("❌ Upload une vidéo d'abord.")
        st.stop()

    # Sauver la vidéo uploadée temporairement
    tmp_path = "uploaded_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Impossible de lire la vidéo.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = time.time()
    proc_filter, conf_thr, red_thr, yellow_thr = pick_params()

    processed = 0
    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_proc = proc_filter(frame)

        # YOLO
        results = model_yolo.predict(frame_proc, conf=float(conf_thr), verbose=False)

        # MiDaS depth
        img_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(DEVICE)
        with torch.no_grad():
            prediction = model_depth(input_batch)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_proc.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        depth_map = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        danger_level = "green"
        pedestrian_count = 0
        min_distance = None

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for b in boxes:
                if int(b.cls[0]) != 0:  # person
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cx = max(0, min(cx, 639))
                cy = max(0, min(cy, 479))

                dval = depth_map[cy, cx]
                distance = 1.0 / (dval + 1e-6)

                pedestrian_count += 1
                min_distance = distance if min_distance is None else min(min_distance, distance)

                in_zone = y2 > int(zone_y)
                if in_zone and distance < float(red_thr):
                    danger_level = "red"
                elif in_zone and distance < float(yellow_thr) and danger_level != "red":
                    danger_level = "yellow"

                # bbox color (global state)
                color = (0, 255, 0)
                if danger_level == "yellow":
                    color = (0, 255, 255)
                elif danger_level == "red":
                    color = (0, 0, 255)

                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_proc, f"{distance:.1f} m", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # zone line + labels
        cv2.line(frame_proc, (0, int(zone_y)), (639, int(zone_y)), (255, 255, 255), 2)
        cv2.putText(frame_proc, f"MODE: {mode} | conf={float(conf_thr):.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if danger_level == "red":
            cv2.putText(frame_proc, "⚠️ PEDESTRIAN DANGER", (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        # counters
        if danger_level == "yellow":
            st.session_state.yellow_frames += 1
        elif danger_level == "red":
            st.session_state.red_frames += 1

        # history
        t = (processed / float(fps))
        st.session_state.hist_t.append(t)
        st.session_state.hist_min_d.append(float(min_distance) if min_distance is not None else np.nan)
        st.session_state.hist_state.append(danger_level)

        # render
        frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        frame_box.image(frame_rgb, channels="RGB", use_container_width=True)

        state_txt = "SAFE" if danger_level == "green" else ("WARNING" if danger_level == "yellow" else "DANGER")
        kpi_box.markdown(
            f"### État ADAS : **{state_txt}**\n"
            f"- Mode : **{mode}**\n"
            f"- Piétons : **{pedestrian_count}**\n"
            f"- Distance min : **{(min_distance if min_distance is not None else float('nan')):.2f} m**\n\n"
            f"**Frames JAUNES** : {st.session_state.yellow_frames}  \n"
            f"**Frames ROUGES** : {st.session_state.red_frames}"
        )

        # charts (last N)
        N = min(250, len(st.session_state.hist_t))
        t_hist = st.session_state.hist_t[-N:]
        d_hist = st.session_state.hist_min_d[-N:]
        s_hist = [state_to_num(x) for x in st.session_state.hist_state[-N:]]

        chart1.line_chart({"min_distance_m": d_hist}, x=t_hist)
        chart2.line_chart({"state(0=G,1=Y,2=R)": s_hist}, x=t_hist)

        processed += 1

    cap.release()
    st.success(f"✅ Analyse terminée : {processed} frames (mode={mode})")

