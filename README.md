# 🤸 Gymnastics Apparatus Scorer

A high-performance AI system for detecting gymnastics apparatus and analyzing athlete performance. The project uses MediaPipe for pose estimation and a custom YOLOv8 model for specialized apparatus detection.

## 📊 Model Performance: YOLOv8m

Our custom-trained YOLOv8 medium model achieves high accuracy in identifying key gymnastics equipment:

| Metric | Value |
| :--- | :--- |
| **mAP50** | 0.7308 |
| **mAP50-95** | 0.4784 |
| **Precision** | 0.7394 |
| **Recall** | 0.6359 |

**Target Classes:** Pommel Horse, Still Rings, Vault, Parallel Bars, Horizontal Bar, Uneven Bars, Balance Beam.

## 🚀 Project Structure

- `api/`: FastAPI backend for specialized analysis logic and scoring.
- `frontend/`: React + Vite application for the judging interface and control view.
- `model_service/`: AI model logic, including:
    - `gymnastics.py`: Core analysis engine using MediaPipe + YOLO.
    - `train_yolov8_colab.py`: Local training script for the apparatus detector.
- `gym_data/`: Dataset storage for training and validation.

## 🛠️ Setup Instructions

### 1. Backend Setup
```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Model Training (Local)
To retrain the apparatus detector on your local GPU:
```bash
python model_service/train_yolov8_colab.py
```
*Requires NVIDIA GPU with CUDA support and the `ultralytics` package.*

## ✨ Features

- **Real-time Apparatus Tracking**: Automatically locks onto equipment using custom YOLOv8.
- **WAG/MAG Scoring**: Automated D-score and E-score contribution analysis.
- **Skeleton Overlay**: High-precision 3D pose visualization for biomechanical analysis.
- **Perspective Calibration**: Auto-calibrates pixel measurements to real-world centimeters based on apparatus dimensions.
