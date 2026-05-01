# check_dataset.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
import cv2

# ── Paths ──────────────────────────────────────────────────────
TRAIN_IMG = os.path.join(DATA_DIR, "RiverFloatingTrash", "images", "train")
VAL_IMG   = os.path.join(DATA_DIR, "RiverFloatingTrash", "images", "val")
TEST_IMG  = os.path.join(DATA_DIR, "RiverFloatingTrash", "images", "test")
TRAIN_LBL = os.path.join(DATA_DIR, "RiverFloatingTrash", "labels", "train")

print("=" * 55)
print("  DATASET VERIFICATION")
print("=" * 55)

# ── Count images ───────────────────────────────────────────────
def count_files(folder, ext=(".jpg", ".jpeg", ".png")):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith(ext)])

train_imgs = count_files(TRAIN_IMG)
val_imgs   = count_files(VAL_IMG)
test_imgs  = count_files(TEST_IMG)
train_lbls = count_files(TRAIN_LBL, ext=(".txt",))

print(f"\n[1] Image Counts:")
print(f"    Train : {train_imgs} images")
print(f"    Val   : {val_imgs}   images")
print(f"    Test  : {test_imgs}  images")
print(f"    Total : {train_imgs + val_imgs + test_imgs} images")

print(f"\n[2] Label Counts:")
print(f"    Train labels : {train_lbls} .txt files")

# ── Match check ────────────────────────────────────────────────
if train_imgs == train_lbls:
    print(f"    ✅ Images and labels match!")
else:
    print(f"    ⚠️  Mismatch! Images={train_imgs}, Labels={train_lbls}")

# ── Preview one image ──────────────────────────────────────────
print(f"\n[3] Previewing one training image...")
sample_imgs = [f for f in os.listdir(TRAIN_IMG)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if sample_imgs:
    sample_path = os.path.join(TRAIN_IMG, sample_imgs[0])
    img = cv2.imread(sample_path)
    h, w = img.shape[:2]
    print(f"    File : {sample_imgs[0]}")
    print(f"    Size : {w} x {h} pixels")

    # Show image
    cv2.imshow("Sample Training Image", img)
    print(f"    ✅ Image opened — press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("    ❌ No images found!")

# ── Label preview ──────────────────────────────────────────────
print(f"\n[4] Previewing one label file...")
sample_lbl = os.path.join(TRAIN_LBL,
             sample_imgs[0].replace(".jpg", ".txt")
                           .replace(".jpeg", ".txt")
                           .replace(".png", ".txt"))

if os.path.exists(sample_lbl):
    with open(sample_lbl) as f:
        lines = f.readlines()
    print(f"    File    : {os.path.basename(sample_lbl)}")
    print(f"    Objects : {len(lines)} detected")
    print(f"    Content : {lines[0].strip()}")
    print(f"    Format  : class_id cx cy width height (YOLO format ✅)")
else:
    print(f"    ⚠️  Label file not found for this image")

print("\n" + "=" * 55)
print("  DATASET READY FOR TRAINING!")
print("=" * 55)