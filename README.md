# AI-Based Real-Time Object Detection and Tracking System

## Overview

This project is a real-time AI-powered object detection and tracking system built using Python, OpenCV, and YOLO models.
The system detects multiple objects from a live webcam/video feed and tracks them in real time with unique IDs.

It is designed for applications such as:

* Smart surveillance
* Traffic monitoring
* Crowd analysis
* Security systems
* Industrial monitoring

---

# Features

* Real-time object detection
* Multi-object tracking
* Unique tracking IDs
* YOLO-based deep learning detection
* ByteTrack / DeepSORT tracking
* Live webcam support
* FPS monitoring
* Bounding boxes and labels
* Optimized for real-time performance

---

# Technologies Used

| Technology           | Purpose                   |
| -------------------- | ------------------------- |
| Python               | Main programming language |
| OpenCV               | Video processing          |
| YOLOv8 / YOLO11      | Object detection          |
| DeepSORT / ByteTrack | Object tracking           |
| NumPy                | Array operations          |

---

# Project Structure

```bash
project/
│
├── main.py
├── advanced_tracking.py
├── ultimate_ai_tracking.py
├── yolov8n.pt
├── yolov8m.pt
├── yolo11m.pt
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/real-time-object-detection.git
```

---

## 2. Open Project Folder

```bash
cd real-time-object-detection
```

---

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# Required Libraries

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install deep-sort-realtime
pip install supervision
```

---

# Running the Project

## Basic Detection

```bash
python main.py
```

---

## Advanced Tracking

```bash
python advanced_tracking.py
```

---

## Ultimate AI Tracking System

```bash
python ultimate_ai_tracking.py
```

---

# How It Works

1. Webcam captures live video
2. YOLO model detects objects
3. Tracking algorithm assigns unique IDs
4. OpenCV displays results in real time

---

# Example Output

```text
person ID:1
chair ID:2
bottle ID:3
```

---

# YOLO Models Used

| Model   | Speed  | Accuracy  |
| ------- | ------ | --------- |
| YOLOv8n | Fast   | Medium    |
| YOLOv8m | Medium | High      |
| YOLO11m | High   | Very High |

---

# Future Improvements

* Face recognition
* Intrusion detection
* Object counting
* Vehicle speed estimation
* Multi-camera support
* AI dashboard
* Database logging
* Fire and smoke detection

---

# Applications

* Smart CCTV systems
* Traffic analysis
* Security monitoring
* Industrial automation
* Smart city surveillance

---

# Performance Optimization

The project includes:

* Resolution optimization
* Lightweight models
* FPS monitoring
* Efficient tracking algorithms

---

# Author

Jatin Kumar

---

# License

This project is for educational and learning purposes.
