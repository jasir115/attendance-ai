---

# 🛡️ AI Face Attendance System with Liveness Detection

A robust, real-time biometric attendance solution utilizing **InsightFace** for high-accuracy recognition and a motion-based **Liveness Detection** protocol to prevent spoofing using photos or videos.

---

## 🌟 Key Features

* **Deep Learning Recognition:** Powered by the `buffalo_l` InsightFace model for high accuracy.
* **Anti-Spoofing Protocol:** Random motion challenges (LEFT, RIGHT, CLOSER) ensure physical presence.
* **Vectorized Matching:** Uses NumPy matrix operations for fast similarity computation.
* **Automatic IN/OUT Logging:** Smart cooldown-based attendance marking.
* **CSV Attendance Records:** Lightweight and easy-to-export logging format.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Computer Vision:** OpenCV, InsightFace (ONNX Runtime)
* **Numerical Processing:** NumPy
* **Data Storage:** CSV files

---

## 💻 Installation

### 1. Environment Setup (Recommended)

Using **Python 3.10** is strongly recommended for compatibility with InsightFace and ONNX Runtime.

```bash
conda create -n attendance python=3.10 -y
conda activate attendance
```

---

### 2. Install Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy insightface onnxruntime
```

> ⚠ If InsightFace models do not download automatically, run the program once with an active internet connection so models can be fetched.

---

## 🚀 How to Use

### Step 1: Register Face (Create Database)

Before recognition, enroll users:

```bash
python register.py
```

* Enter the user's name
* Move head slightly while samples are captured
* This generates or updates:

```
db/embeddings.npz
```

---

### Step 2: Start Attendance System

Run the main attendance script:

```bash
python attendance.py
```

The system will:

* Detect the face in real time
* Ask for random motion challenge (LEFT / RIGHT / CLOSER)
* Verify liveness
* Recognize identity
* Mark **IN / OUT** automatically

Attendance records are saved to:

```
attendance.csv
```

---

## 🔐 Liveness Detection Logic

To prevent spoofing using printed photos or mobile screens, the system performs:

* Horizontal head movement detection
* Face scale change (moving closer to camera)

Recognition is only unlocked after successful liveness verification.

---

## 📂 Project Structure

```
attendance_ai/
├── db/
│   └── embeddings.npz        # Generated after registration
├── attendance.py             # Real-time recognition & attendance
├── register.py               # Face enrollment
├── attendance.csv            # Auto-generated attendance log
├── requirements.txt
└── README.md
```

> Model files and biometric embeddings are excluded from GitHub for privacy and security.

---

## 🚧 Future Improvements

* CNN-based deep anti-spoofing model
* Web dashboard for attendance monitoring
* Admin panel for user management
* Cloud-based attendance storage
* Multi-camera support

---

## ⚠ Disclaimer

This project is intended for educational and research purposes.
For production deployment, stronger anti-spoofing models and data security mechanisms are required.

---
