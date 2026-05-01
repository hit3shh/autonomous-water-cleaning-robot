# src/dashboard.py
import streamlit as st
import cv2
import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import load_model, detect_objects
from rl_agent  import RLAgent
from config    import DASH_TITLE, CLASS_NAMES

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title = DASH_TITLE,
    page_icon  = "🤖",
    layout     = "wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .action-box {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border: 2px solid #52b788;
    }
    .title-style {
        font-size: 28px;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────
st.markdown(
    '<p class="title-style">🤖 Autonomous Water Surface Cleaning Simulation</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Session state init ─────────────────────────────────────────
defaults = {
    "running"     : False,
    "trash_count" : 0,
    "total_reward": 0.0,
    "step_count"  : 0,
    "history"     : [],
    "class_counts": {c: 0 for c in CLASS_NAMES},
    "last_action" : "IDLE",
    "last_state"  : "No Trash",
    "last_reward" : 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load model + agent ONCE ────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_agent():
    return RLAgent()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    start_btn = st.button("▶️  START", type="primary")
    stop_btn  = st.button("⏹️  STOP")

    st.markdown("---")
    st.markdown("## 📊 Session Stats")
    stat_steps  = st.empty()
    stat_reward = st.empty()
    stat_trash  = st.empty()

    st.markdown("---")
    st.markdown("## 🗂️ Class Detections")
    class_display = st.empty()

    st.markdown("---")
    if st.button("🔄 Reset Stats"):
        st.session_state.trash_count  = 0
        st.session_state.total_reward = 0.0
        st.session_state.step_count   = 0
        st.session_state.history      = []
        st.session_state.class_counts = {c: 0 for c in CLASS_NAMES}

    st.markdown("---")
    st.markdown("""
    ### ℹ️ Actions
    - ⬅️ **LEFT**
    - ➡️ **RIGHT**
    - ⬆️ **FORWARD**
    - ⬇️ **BACKWARD**
    - 🤖 **COLLECT**
    """)

# ── Layout ─────────────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2])

with col_cam:
    st.markdown("### 📷 Live Camera Feed")
    camera_placeholder = st.empty()

with col_info:
    st.markdown("### 🤖 Robot Decision")
    action_placeholder = st.empty()
    st.markdown("### 📦 Detections")
    detection_placeholder = st.empty()
    st.markdown("### 📈 Reward History")
    chart_placeholder = st.empty()

# ── Action styling ─────────────────────────────────────────────
ACTION_EMOJI = {
    "LEFT"    : "⬅️  LEFT",
    "RIGHT"   : "➡️  RIGHT",
    "FORWARD" : "⬆️  FORWARD",
    "BACKWARD": "⬇️  BACKWARD",
    "COLLECT" : "🤖  COLLECT",
    "IDLE"    : "💤  IDLE"
}
ACTION_COLOR = {
    "LEFT"    : "#4fc3f7",
    "RIGHT"   : "#ef5350",
    "FORWARD" : "#66bb6a",
    "BACKWARD": "#ffa726",
    "COLLECT" : "#ffee58",
    "IDLE"    : "#90a4ae"
}

# ── Button logic ───────────────────────────────────────────────
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ── Helper: draw reward chart ──────────────────────────────────
def draw_chart(history):
    if len(history) < 2:
        return
    max_val = max(abs(v) for v in history) or 1
    bars = ""
    for v in history[-40:]:
        pct   = int((v + max_val) / (2 * max_val) * 100)
        color = "#66bb6a" if v >= 0 else "#ef5350"
        h_px  = max(4, pct)
        bars += (f'<div style="display:inline-block;width:7px;'
                 f'height:{h_px}px;background:{color};'
                 f'margin:1px;vertical-align:bottom"></div>')
    chart_placeholder.markdown(
        f'<div style="background:#1e2130;padding:10px;'
        f'border-radius:8px;height:100px;'
        f'display:flex;align-items:flex-end">{bars}</div>'
        f'<small style="color:#aaa">🟢 +reward &nbsp; 🔴 -reward</small>',
        unsafe_allow_html=True
    )

# ── IDLE screen ────────────────────────────────────────────────
def show_idle():
    camera_placeholder.markdown("""
    <div style="background:#1e2130;border-radius:12px;
                padding:80px;text-align:center;
                border:2px dashed #3d4470">
        <h2 style="color:#00d4ff">📷 Camera Idle</h2>
        <p style="color:#aaa">Press ▶️ START to begin</p>
    </div>
    """, unsafe_allow_html=True)

    action_placeholder.markdown("""
    <div class="action-box" style="color:#888;border-color:#444">
        💤 IDLE<br>
        <small style="color:#666">Waiting to start...</small>
    </div>
    """, unsafe_allow_html=True)
    detection_placeholder.markdown("*Press START to begin*")

# ── MAIN LOOP ──────────────────────────────────────────────────
if st.session_state.running:

    model = get_model()
    agent = get_agent()

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW = more stable on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        st.error("❌ Webcam not found! Check if camera is connected.")
        st.stop()

    st.success("✅ Camera running — press STOP to end session")

    frame_fail = 0  # count consecutive failures

    while st.session_state.running:

        ret, frame = cap.read()

        # Handle read failures gracefully
        if not ret or frame is None:
            frame_fail += 1
            if frame_fail > 10:
                st.error("❌ Camera disconnected!")
                break
            time.sleep(0.1)
            continue

        frame_fail = 0  # reset on success
        h, w = frame.shape[:2]

        # ── Detect ────────────────────────────────────────────
        annotated, detections = detect_objects(model, frame)

        # ── RL step ───────────────────────────────────────────
        action, info = agent.step(detections, w, h)

        # ── Update stats ──────────────────────────────────────
        st.session_state.step_count   = info["step"]
        st.session_state.total_reward = info["total_reward"]
        st.session_state.last_action  = action
        st.session_state.last_state   = info["state_label"]
        st.session_state.last_reward  = info["reward"]

        if action == "COLLECT" and detections:
            if info["state_label"] == "Center":
                st.session_state.trash_count += len(detections)

        for d in detections:
            name = d["class_name"]
            if name in st.session_state.class_counts:
                st.session_state.class_counts[name] += 1

        st.session_state.history.append(info["reward"])
        if len(st.session_state.history) > 50:
            st.session_state.history.pop(0)

        # ── Camera feed ───────────────────────────────────────
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(
            rgb, channels="RGB",
            use_container_width=True
        )

        # ── Action box ────────────────────────────────────────
        color        = ACTION_COLOR.get(action, "#ffffff")
        emoji_action = ACTION_EMOJI.get(action, action)
        action_placeholder.markdown(f"""
        <div class="action-box"
             style="border-color:{color};color:{color}">
            {emoji_action}<br>
            <small style="color:#aaa">
                📍 {info['state_label']}<br>
                💰 Reward: {info['reward']} &nbsp;|&nbsp;
                🔢 Step: {info['step']}
            </small>
        </div>
        """, unsafe_allow_html=True)

        # ── Detection list ────────────────────────────────────
        if detections:
            det_md = ""
            for d in detections[:5]:
                bar = "█" * int(d['confidence'] * 10)
                det_md += (f"**{d['class_name'].upper()}**\n"
                           f"- Conf: `{d['confidence']:.0%}` {bar}\n"
                           f"- Pos: `{d['center']}`\n\n---\n")
            detection_placeholder.markdown(det_md)
        else:
            detection_placeholder.info("🔍 No trash detected")

        # ── Sidebar stats ─────────────────────────────────────
        stat_steps.markdown(
            f"**Steps:** `{st.session_state.step_count}`")
        stat_reward.markdown(
            f"**Total Reward:** `{st.session_state.total_reward}`")
        stat_trash.markdown(
            f"**Trash Collected:** `{st.session_state.trash_count}`")

        cls_md = ""
        for cls, cnt in st.session_state.class_counts.items():
            if cnt > 0:
                cls_md += f"- **{cls}**: {cnt}\n"
        if cls_md:
            class_display.markdown(cls_md)

        # ── Chart ─────────────────────────────────────────────
        draw_chart(st.session_state.history)

        time.sleep(0.05)

    # Cleanup
    cap.release()
    agent.save_q_table()
    st.info("⏹️ Session stopped. Q-table saved.")

else:
    show_idle()