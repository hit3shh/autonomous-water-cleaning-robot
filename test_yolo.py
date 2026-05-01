# test_yolo.py  — run once to confirm YOLO works, then delete
from ultralytics import YOLO
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import YOLO_MODEL, MODELS_DIR

print("=" * 50)
print("  YOLOv8 SETUP TEST")
print("=" * 50)

# Step 1 — Download yolov8n.pt into models/
print("\n[1] Loading YOLOv8n model (downloads if not present)...")
model = YOLO("yolov8n.pt")  # downloads automatically

# Step 2 — Move it to models/ folder
import shutil
if os.path.exists("yolov8n.pt"):
    shutil.move("yolov8n.pt", YOLO_MODEL)
    print(f"    ✅ Model saved to: {YOLO_MODEL}")
else:
    print(f"    ✅ Model already at: {YOLO_MODEL}")

# Step 3 — Reload from models/ and check
model = YOLO(YOLO_MODEL)
print(f"\n[2] Model loaded successfully!")
print(f"    Task   : {model.task}")
print(f"    Classes: {len(model.names)} (COCO default)")

# Step 4 — GPU check
import torch
print(f"\n[3] Device Check:")
print(f"    CUDA Available : {torch.cuda.is_available()}")
print(f"    GPU            : {torch.cuda.get_device_name(0)}")
print(f"    YOLO will use  : {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

print("\n" + "=" * 50)
print("  ALL CHECKS PASSED — Ready for Step 4!")
print("=" * 50)