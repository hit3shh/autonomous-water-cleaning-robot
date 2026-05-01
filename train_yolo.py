# train_yolo.py
import os, sys, shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RFT_YAML, YOLO_MODEL, MODELS_DIR
from ultralytics import YOLO
import torch

def main():
    print("=" * 55)
    print("  YOLOv8 TRAINING — RiverFloatingTrash Dataset")
    print("=" * 55)

    # ── Confirm GPU ────────────────────────────────────────
    print(f"\n[1] Device  : {'GPU ✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ⚠️'}")
    print(f"[2] Dataset : {RFT_YAML}")
    print(f"[3] Model   : {YOLO_MODEL}")

    # ── Load base model ────────────────────────────────────
    model = YOLO(YOLO_MODEL)

    print("\n[4] Starting training...")
    print("    This will take 15-30 minutes on RTX 2050")
    print("    You will see progress per epoch below\n")

    # ── Train ──────────────────────────────────────────────
    results = model.train(
        data=RFT_YAML,
        epochs=50,
        imgsz=416,
        batch=8,
        device=0,
        workers=0,        # ← Fix for Windows multiprocessing
        project="runs",
        name="trash_detect",
        exist_ok=True,
        patience=10,
        save=True,
        plots=False,
        verbose=True
    )

    # ── Copy best model to models/ ─────────────────────────
    best_src = os.path.join("runs", "detect", "trash_detect", "weights", "best.pt")
    best_dst = os.path.join(MODELS_DIR, "best.pt")

    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"\n✅ Best model saved to: {best_dst}")
    else:
        print(f"\n⚠️  Searching for best.pt...")
        for root, dirs, files in os.walk("runs"):
            for file in files:
                if file == "best.pt":
                    found = os.path.join(root, file)
                    shutil.copy(found, best_dst)
                    print(f"✅ Found and saved: {found}")
                    break

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE — Ready for Step 6!")
    print("=" * 55)

# ── REQUIRED for Windows ───────────────────────────────────────
if __name__ == '__main__':
    main()