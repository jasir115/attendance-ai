# 🛡️ AI Face Attendance System with Liveness Detection

A robust, real-time biometric attendance solution utilizing **InsightFace** for high-accuracy recognition and a motion-based **Liveness Detection** protocol to prevent spoofing from photos or videos.



## 🌟 Key Features
* **Deep Learning Recognition:** Powered by the state-of-the-art `buffalo_l` model for enterprise-grade accuracy.
* **Anti-Spoofing Protocol:** Random motion challenges (LEFT, RIGHT, CLOSER) ensure the user is physically present.
* **Vectorized Search:** Uses NumPy-based matrix operations for near-instant matching.
* **Smart Logging:** Automatically manages `IN/OUT` status with a configurable cooldown period.
* **Admin Dashboard:** Secure interface to view or remove registered users.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Computer Vision:** OpenCV, InsightFace (ONNX Runtime)
* **Data Processing:** NumPy, CSV
* **Interface:** Tkinter

---

## 💻 Installation

### 1. Environment Setup (Recommended)
Using **Python 3.10** is critical for compatibility with ONNX and InsightFace libraries.

```bash
# Create the environment
conda create -n attendance python=3.10 -y

# Activate the environment
conda activate attendance
2. Install Dependencies
Run the following command to install all necessary libraries:

Bash
pip install opencv-python numpy insightface onnxruntime-gpu
Note: If you do not have an NVIDIA GPU, install the standard CPU version instead:

pip install onnxruntime

🚀 How to Use
Step 1: Face Registration
Before the system can recognize you, you must enroll your face into the database.

Bash
python register.py
Enter the user's name.

Follow the on-screen prompts to capture facial samples.

This creates or updates the db/embeddings.npz file.

Step 2: Start Attendance
Launch the main application to begin the monitoring process.

Bash
python attendance.py
Click "Start Attendance".

Liveness Check: The system will issue a random command (e.g., "DO: LEFT"). Move your head accordingly.

Verification: Once liveness is confirmed, the system unlocks the recognition engine.

Logging: Validated entries are saved instantly to attendance.csv.

📂 Project Structure
attendance_ai/
├── db/                   # Database for facial embeddings
│   └── embeddings.npz    
├── attendance.py         # Main application & GUI logic
├── register.py           # User enrollment & sample capture
├── attendance.csv        # Auto-generated logs
├── requirements.txt      # Project dependencies
└── README.md             # Documentation