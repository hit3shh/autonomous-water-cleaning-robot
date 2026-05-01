# src/detection.py
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINED_MODEL, CONFIDENCE, CLASS_NAMES
from ultralytics import YOLO

# ── Load Model ─────────────────────────────────────────────────
def load_model():
    """Load trained YOLOv8 model."""
    if not os.path.exists(TRAINED_MODEL):
        print(f"❌ Trained model not found at {TRAINED_MODEL}")
        print("   Run train_yolo.py first!")
        sys.exit(1)
    model = YOLO(TRAINED_MODEL)
    print(f"✅ Model loaded: {TRAINED_MODEL}")
    return model

# ── Detection Function ─────────────────────────────────────────
def detect_objects(model, frame):
    """
    Run YOLO detection on a single frame.
    Returns:
        - annotated frame with bounding boxes
        - list of detected objects with details
    """
    detections = []

    # Run YOLO inference
    results = model(frame, conf=CONFIDENCE, verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Get coordinates (x1,y1 = top-left, x2,y2 = bottom-right)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get confidence and class
            confidence = float(box.conf[0])
            class_id   = int(box.cls[0])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"

            detections.append({
                "class_id"  : class_id,
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "bbox"      : (x1, y1, x2, y2),
                "center"    : (cx, cy)
            })

            # ── Draw bounding box ──────────────────────────────
            color = get_class_color(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Draw label ─────────────────────────────────────
            label = f"{class_name} {confidence:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame,
                          (x1, y1 - label_size[1] - 8),
                          (x1 + label_size[0], y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # ── Draw center dot ────────────────────────────────
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame, detections

# ── Color per class ────────────────────────────────────────────
def get_class_color(class_id):
    """Return a unique BGR color for each class."""
    colors = [
        (0, 255, 0),    # bottle      — green
        (255, 165, 0),  # grass       — orange
        (255, 0, 0),    # branch      — blue
        (0, 255, 255),  # milk-box    — yellow
        (255, 0, 255),  # plastic-bag — magenta
        (0, 128, 255),  # plastic-garbage — orange-blue
        (128, 0, 255),  # ball        — purple
        (0, 255, 128),  # leaf        — mint
    ]
    return colors[class_id % len(colors)]

# ── Webcam Test ────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()
    cap   = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("❌ Webcam not found!")
        sys.exit(1)

    print("✅ Webcam opened — press Q to quit")
    print("   Point camera at trash objects to detect them")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Run detection
        annotated_frame, detections = detect_objects(model, frame)

        # Show trash count on screen
        count = len(detections)
        cv2.putText(annotated_frame,
                    f"Trash detected: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Print detections to terminal
        for d in detections:
            print(f"  [{d['class_name']}] conf={d['confidence']} center={d['center']}")

        # Show frame
        cv2.imshow("Water Cleaning Robot - Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detection stopped")