# src/main.py
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import load_model, detect_objects
from rl_agent  import RLAgent
from config    import CONFIDENCE

# ── Action display colors ──────────────────────────────────────
ACTION_COLORS = {
    "LEFT"     : (255, 0,   0  ),   # Blue
    "RIGHT"    : (0,   0,   255),   # Red
    "FORWARD"  : (0,   255, 0  ),   # Green
    "BACKWARD" : (0,   165, 255),   # Orange
    "COLLECT"  : (0,   255, 255),   # Yellow
}

# ── Arrow directions for actions ───────────────────────────────
ACTION_ARROWS = {
    "LEFT"    : "⬅",
    "RIGHT"   : "➡",
    "FORWARD" : "⬆",
    "BACKWARD": "⬇",
    "COLLECT" : "🤖",
}

def draw_hud(frame, detections, action, info, trash_count):
    """Draw heads-up display on the frame."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ───────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # ── Title ──────────────────────────────────────────────────
    cv2.putText(frame, "WATER CLEANING ROBOT SIMULATION",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # ── Trash count ────────────────────────────────────────────
    cv2.putText(frame, f"Trash Detected: {len(detections)}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    # ── Total collected ────────────────────────────────────────
    cv2.putText(frame, f"Total Count: {trash_count}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)

    # ── Action box (bottom right) ──────────────────────────────
    action_color = ACTION_COLORS.get(action, (255, 255, 255))
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (w-220, h-100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "ROBOT ACTION:",
                (w-210, h-75), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(frame, action,
                (w-210, h-40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, action_color, 3)

    # ── State info (bottom left) ───────────────────────────────
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h-100), (280, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Position: {info['state_label']}",
                (10, h-75), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Reward: {info['reward']}  Total: {info['total_reward']}",
                (10, h-45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Step: {info['step']}",
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    return frame


def main():
    print("=" * 55)
    print("  WATER CLEANING ROBOT — FULL PIPELINE")
    print("=" * 55)

    # ── Load YOLO model ────────────────────────────────────────
    print("\n[1] Loading detection model...")
    model = load_model()

    # ── Load RL Agent ──────────────────────────────────────────
    print("[2] Loading RL agent...")
    agent = RLAgent()

    # ── Open webcam ────────────────────────────────────────────
    print("[3] Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not found!")
        sys.exit(1)

    print("\n✅ System ready! Press Q to quit\n")
    print(f"{'Step':<6} {'Object':<18} {'Position':<16} {'Action':<10} {'Reward'}")
    print("-" * 65)

    trash_count  = 0
    frame_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Step 1: Detect trash ───────────────────────────────
        annotated_frame, detections = detect_objects(model, frame)

        # ── Step 2: RL agent decides action ───────────────────
        action, info = agent.step(detections, w, h)

        # ── Step 3: Count collected trash ─────────────────────
        if action == "COLLECT" and len(detections) > 0:
            if info["state_label"] == "Center":
                trash_count += len(detections)

        # ── Step 4: Draw HUD on frame ─────────────────────────
        annotated_frame = draw_hud(
            annotated_frame, detections,
            action, info, trash_count
        )

        # ── Step 5: Print to terminal (every 10 frames) ────────
        if frame_count % 10 == 0:
            obj_name = detections[0]["class_name"] if detections else "none"
            print(f"{info['step']:<6} {obj_name:<18} "
                  f"{info['state_label']:<16} {action:<10} {info['reward']}")

        # ── Step 6: Show frame ─────────────────────────────────
        cv2.imshow("Water Cleaning Robot", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ────────────────────────────────────────────────
    agent.save_q_table()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print(f"  SESSION COMPLETE")
    print(f"  Total steps    : {info['step']}")
    print(f"  Total reward   : {info['total_reward']}")
    print(f"  Trash collected: {trash_count}")
    print("=" * 55)


# ── Required for Windows ───────────────────────────────────────
if __name__ == "__main__":
    main()