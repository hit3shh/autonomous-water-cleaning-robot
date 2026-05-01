# 🤖 Autonomous Water Surface Cleaning Simulation

A software-only simulation dashboard that uses **YOLOv8** for real-time 
trash detection and **Q-Learning** for autonomous robot decision making.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Flask](https://img.shields.io/badge/Flask-3.x-red)
![CUDA](https://img.shields.io/badge/CUDA-RTX_2050-yellow)

---

## 📌 Project Overview

| Component        | Technology              |
|------------------|-------------------------|
| Object Detection | YOLOv8n (Ultralytics)   |
| Computer Vision  | OpenCV                  |
| RL Algorithm     | Q-Learning (NumPy)      |
| Web Dashboard    | Flask + HTML/CSS/JS     |
| GPU              | NVIDIA RTX 2050 (CUDA)  |
| Dataset          | RiverFloatingTrash      |

---

## 🎯 Features

- ✅ Real-time trash detection using trained YOLOv8 model
- ✅ Q-Learning robot that learns optimal collection strategy
- ✅ Professional Flask dashboard with live camera feed
- ✅ Detection history with timestamps
- ✅ Reward trend chart
- ✅ Per-class detection statistics
- ✅ GPU accelerated inference

---

## 📁 Project Structure
water_cleaning_sim/
├── config.py                 # Global settings
├── train_yolo.py             # YOLOv8 training script
├── flask_app/
│   ├── app.py                # Flask server
│   ├── templates/
│   │   └── index.html        # Dashboard UI
│   └── static/
│       ├── css/style.css     # Styling
│       └── js/dashboard.js   # Real-time updates
├── src/
│   ├── detection.py          # YOLO detection module
│   ├── rl_agent.py           # Q-Learning agent
│   └── main.py               # Terminal pipeline
├── models/
│   ├── yolov8n.pt            # Base model
│   └── best.pt               # Trained model
├── data/
│   └── RiverFloatingTrash/   # Dataset
├── logs/
│   └── q_table.npy           # Saved Q-table
└── requirements.txt

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/water_cleaning_sim.git
cd water_cleaning_sim
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Train the model (optional — best.pt already included)
```bash
python train_yolo.py
```

### 6. Run the dashboard
```bash
python flask_app/app.py
```

### 7. Open browser
http://127.0.0.1:5000

---

## 📊 Training Results

| Metric           | Value  |
|------------------|--------|
| Dataset          | RiverFloatingTrash (2400 images) |
| Classes          | 8 (bottle, grass, branch, etc.) |
| Epochs           | 49/50 (early stopped) |
| mAP50            | 72.5%  |
| Best class       | ball (95.2%) |
| Training time    | ~27 min on RTX 2050 |

---

## 🧠 RL Agent Details

| Parameter  | Value           |
|------------|-----------------|
| Algorithm  | Q-Learning      |
| States     | 10 (3x3 grid + no-trash) |
| Actions    | 5 (LEFT, RIGHT, FORWARD, BACKWARD, COLLECT) |
| Alpha (LR) | 0.1             |
| Gamma      | 0.9             |
| Epsilon    | 0.2 (exploration) |

---

## ⚙️ System Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Webcam
- 8GB+ RAM

---

## 👤 Author

Made with ❤️ as a minor project