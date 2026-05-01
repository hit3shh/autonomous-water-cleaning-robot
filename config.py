# config.py
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
RUNS_DIR    = os.path.join(BASE_DIR, "runs")

# Dataset
RFT_YAML    = os.path.join(DATA_DIR, "RiverFloatingTrash", "RFT.yaml")

# Model
YOLO_MODEL  = os.path.join(MODELS_DIR, "yolov8n.pt")   # pretrained base
TRAINED_MODEL = os.path.join(MODELS_DIR, "best.pt")    # after training

# ── Detection Settings ─────────────────────────────────
CONFIDENCE  = 0.4        # minimum confidence to show detection
IMG_SIZE    = 640        # YOLO input size

# ── RL Agent Settings ──────────────────────────────────
GRID_COLS   = 3          # divide frame into 3 columns: LEFT, CENTER, RIGHT
GRID_ROWS   = 3          # divide frame into 3 rows: TOP, MIDDLE, BOTTOM
Q_TABLE_PATH = os.path.join(LOGS_DIR, "q_table.npy")

# ── Dashboard ──────────────────────────────────────────
DASH_TITLE  = "Autonomous Water Surface Cleaning Simulation"

# Update this line in config.py
RFT_YAML = os.path.join(DATA_DIR, "RiverFloatingTrash", "RFT.yaml")

# Also add class names
CLASS_NAMES = ['bottle', 'grass', 'branch', 'milk-box',
               'plastic-bag', 'plastic-garbage', 'ball', 'leaf']
NUM_CLASSES = 8