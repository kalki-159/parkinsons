# CogniVue - Parkinson's Disease Detection System

AI-powered Parkinson's disease screening using video hand tremor analysis with MediaPipe and machine learning.

---

## Project Overview

**Project Title:** CogniVue - AI-Based Parkinson's Disease Detection System

**Objective:** To develop a non-invasive screening tool that detects potential Parkinson's disease symptoms through video-based hand tremor analysis.

**Methodology:**
1. Capture hand movements via webcam or video upload
2. Extract hand landmarks using MediaPipe
3. Analyze tremor patterns (frequency, amplitude, rhythm)
4. Calculate risk score using rule-based analysis

---

## Features

| Feature | Description |
|---------|-------------|
| **Live Camera Analysis** | Real-time hand detection and tremor analysis |
| **Video Upload** | Upload pre-recorded videos for analysis |
| **Hand Detection** | MediaPipe-powered hand landmark detection |
| **Tremor Analysis** | FFT-based frequency analysis |
| **Risk Assessment** | Rule-based severity classification |
| **Interactive Dashboard** | Visual results with charts and metrics |

---

## Tech Stack

### Backend
- **FastAPI** - Web framework
- **Python 3.10+** - Programming language
- **TensorFlow** - ML model inference
- **MediaPipe** - Hand landmark detection
- **OpenCV** - Video processing
- **NumPy/SciPy** - Signal processing

### Frontend
- **HTML5** - Page structure
- **CSS3** - Styling and animations
- **JavaScript** - Interactivity
- **SVG** - Charts and visualizations

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Webcam (for live analysis)
- Modern web browser (Chrome, Firefox, Edge)

### Steps

1. **Clone or download the project**
   ```bash
   cd parkinson
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

---

## How to Use

### Option 1: Live Camera Analysis

1. Navigate to **Analysis** tab
2. Click **Live Video Feed**
3. Click **Start Camera**
4. Hold your hands steady in front of the camera (5-15 seconds)
5. Click **Record** to start recording
6. Click **Stop Recording** when done
7. Click **Analyse Recording**
8. View results on the dashboard

### Option 2: Video Upload

1. Navigate to **Analysis** tab
2. Click **Upload Video** tab
3. Drag & drop or browse to select a video file
4. Click **Analyse Video**
5. View results on the dashboard

### Recording Tips

| Tip | Description |
|-----|-------------|
| **Hand Visibility** | Keep both hands clearly visible in frame |
| **Lighting** | Ensure good lighting on hands |
| **Distance** | Keep hands 20-40cm from camera |
| **Duration** | Record 5-15 seconds of movement |
| **Position** | Hold hands steady (resting position) |

---

## Results Explained

### Risk Categories

| Category | Score Range | Description |
|----------|-------------|-------------|
| **Normal** | 0-25% | No significant tremor detected |
| **Early Stage** | 25-45% | Mild indicators - monitor symptoms |
| **Moderate** | 45-65% | Consult a healthcare professional |
| **Severe** | >65% | Seek immediate medical consultation |

### Metrics Explained

| Metric | Description | Normal Range |
|--------|-------------|-------------|
| **Tremor Frequency** | Oscillation rate in Hz | <4 Hz (normal) |
| **Amplitude** | Movement deviation in mm | <2 mm (normal) |
| **Rhythm Consistency** | Pattern regularity % | <50% (normal) |
| **Movement Smoothness** | Jerk-free motion % | >85% (normal) |

> **Disclaimer:** This tool is for screening purposes only. It does NOT provide a medical diagnosis. Always consult a healthcare professional for proper evaluation.

---

## Project Structure

```
parkinson/
|-- app.py                  # FastAPI backend server
|-- extract_features.py      # Hand detection & tremor analysis
|-- model_inference.py      # ML model inference & risk calculation
|-- hand_landmarker.task    # MediaPipe hand model
|-- model.h5               # TensorFlow/Keras model
|-- scaler.pkl             # Feature scaler
|-- label_encoder.pkl      # Label encoder
|-- requirements.txt        # Python dependencies
|-- index-full.html         # Main web interface
|-- README.md              # Project documentation
```

---

## How It Works

### 1. Hand Detection
```
Video Frame -> MediaPipe Hands -> 21 Hand Landmarks -> Track wrist position over time
```

### 2. Tremor Analysis
```
Wrist Position (Y-axis) -> FFT Analysis -> Tremor Frequency (Hz)
                         -> Amplitude Calculation
                         -> Rhythm Consistency
                         -> Movement Smoothness
```

### 3. Risk Calculation
```
Features -> Rule-Based Analysis -> Risk Score (0-100%)
                             -> Risk Category
                             -> Recommendations
```

---

## Team

| Name | Role | Contribution |
|------|------|--------------|
| **Sai Vaishnavi S** | AI/ML Lead | Model design & training |
| **P Sahithi** | Full-Stack Developer | Web interface & API |
| **A Veda Reddy** | Computer Vision Engineer | Video processing pipeline |

---

## References

- MediaPipe Hands: [Google AI Blog](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
- Parkinson's Tremor Analysis: [NIH - Tremor](https://www.ninds.nih.gov/Disorders/All-Disorders/Tremor-Information-Page)
- Hoehn & Yahr Staging: [Movement Disorder Society](https://www.movementdisorders.org/)

---

## License

This project is developed for **educational purposes** as a minor project.

> **Important:** This is NOT a medical device and should NOT be used for self-diagnosis. Always consult qualified healthcare professionals.

---

## Acknowledgments

- MediaPipe by Google
- TensorFlow by Google Brain Team
- FastAPI by Sebastian Ramirez

---

**Built for healthcare innovation**

**Version:** 2.0
**Last Updated:** April 2026
