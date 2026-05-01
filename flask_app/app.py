# flask_app/app.py
import sys
import os
import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify
from collections import deque
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection import load_model, detect_objects
from src.rl_agent  import RLAgent
from config        import CLASS_NAMES

# ── Flask app ──────────────────────────────────────────────────
app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  GLOBAL STATE — shared between camera thread and Flask routes
# ══════════════════════════════════════════════════════════════
state = {
    "action"        : "IDLE",
    "state_label"   : "No Trash",
    "reward"        : 0,
    "total_reward"  : 0.0,
    "step"          : 0,
    "trash_count"   : 0,
    "detected_now"  : 0,
    "detections"    : [],
    "class_counts"  : {c: 0 for c in CLASS_NAMES},
    "history"       : deque(maxlen=10),   # last 10 detections
    "reward_trend"  : deque(maxlen=30),   # last 30 rewards for chart
    "running"       : False,
    "fps"           : 0,
}

# Thread lock — prevents data corruption when multiple
# threads read/write state at the same time
lock = threading.Lock()

# ── Load model + agent once at startup ────────────────────────
print("[INFO] Loading YOLO model...")
model = load_model()
print("[INFO] Loading RL agent...")
agent = RLAgent()
print("[INFO] Ready!")

# ══════════════════════════════════════════════════════════════
#  CAMERA THREAD — runs detection + RL in background
# ══════════════════════════════════════════════════════════════
output_frame = None   # latest annotated frame (shared)

def camera_loop():
    """
    Runs in background thread.
    Captures frames → runs YOLO → runs RL → updates state.
    """
    global output_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("[ERROR] Webcam not found!")
        return

    print("[INFO] Camera started")
    fps_counter = 0
    fps_time    = time.time()

    while True:
        if not state["running"]:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # ── Detection ──────────────────────────────────────────
        annotated, detections = detect_objects(model, frame)

        # ── RL decision ────────────────────────────────────────
        action, info = agent.step(detections, w, h)

        # ── FPS calculation ────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time    = time.time()
        else:
            fps = state["fps"]

        # ── Update history if trash detected ───────────────────
        history_entry = None
        if detections:
            obj = detections[0]
            history_entry = {
                "time"      : datetime.now().strftime("%H:%M:%S"),
                "object"    : obj["class_name"],
                "confidence": f"{obj['confidence']:.0%}",
                "action"    : action,
                "reward"    : info["reward"],
            }

        # ── Update global state (thread-safe) ──────────────────
        with lock:
            state["action"]       = action
            state["state_label"]  = info["state_label"]
            state["reward"]       = info["reward"]
            state["total_reward"] = info["total_reward"]
            state["step"]         = info["step"]
            state["detected_now"] = len(detections)
            state["fps"]          = fps
            state["detections"]   = [
                {
                    "class_name" : d["class_name"],
                    "confidence" : d["confidence"],
                    "center"     : d["center"],
                }
                for d in detections[:5]
            ]
            # Update class counts
            for d in detections:
                n = d["class_name"]
                if n in state["class_counts"]:
                    state["class_counts"][n] += 1

            # Collect trash
            if action == "COLLECT" and info["state_label"] == "Center":
                state["trash_count"] += len(detections)

            # Add to history
            if history_entry:
                state["history"].appendleft(history_entry)

            # Add to reward trend
            state["reward_trend"].append(info["reward"])

        # ── Draw FPS on frame ──────────────────────────────────
        cv2.putText(annotated,
                    f"FPS: {fps}",
                    (w - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # ── Store annotated frame for streaming ────────────────
        _, buffer = cv2.imencode(".jpg", annotated,
                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
        output_frame = buffer.tobytes()

        time.sleep(0.03)   # ~30 FPS cap

    cap.release()


# ── Start camera thread ────────────────────────────────────────
camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/start")
def start():
    """Start the detection pipeline."""
    with lock:
        state["running"] = True
    return jsonify({"status": "started"})


@app.route("/stop")
def stop():
    """Stop the detection pipeline."""
    with lock:
        state["running"] = False
    agent.save_q_table()
    return jsonify({"status": "stopped"})


@app.route("/reset")
def reset():
    """Reset all statistics."""
    with lock:
        state["trash_count"]  = 0
        state["total_reward"] = 0.0
        state["step"]         = 0
        state["detected_now"] = 0
        state["detections"]   = []
        state["class_counts"] = {c: 0 for c in CLASS_NAMES}
        state["history"].clear()
        state["reward_trend"].clear()
    return jsonify({"status": "reset"})


@app.route("/status")
def status():
    """Return current system state as JSON — polled by JavaScript."""
    with lock:
        return jsonify({
            "action"       : state["action"],
            "state_label"  : state["state_label"],
            "reward"       : state["reward"],
            "total_reward" : state["total_reward"],
            "step"         : state["step"],
            "trash_count"  : state["trash_count"],
            "detected_now" : state["detected_now"],
            "fps"          : state["fps"],
            "running"      : state["running"],
            "detections"   : state["detections"],
            "class_counts" : state["class_counts"],
            "reward_trend" : list(state["reward_trend"]),
        })


@app.route("/history")
def history():
    """Return detection history as JSON."""
    with lock:
        return jsonify(list(state["history"]))


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream — displays live camera feed in browser."""
    def generate():
        while True:
            if output_frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + output_frame +
                   b"\r\n")
            time.sleep(0.033)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  WATER CLEANING ROBOT — FLASK DASHBOARD")
    print("  Open browser at: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, threaded=True, port=5000)