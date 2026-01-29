import os
os.environ["INSIGHTFACE_HOME"] = r"E:\attendance_ai\insightface_models"

import cv2
import numpy as np
import time
import csv
import random
from insightface.app import FaceAnalysis

# ================= FACE DATABASE =================
data = np.load("db/embeddings.npz", allow_pickle=True)
names = data["names"]
db_embeddings = np.array(data["embeddings"])

# normalize DB embeddings once
db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

# ================= INSIGHTFACE =================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640,640))

# ================= LIVENESS CHALLENGE =================
challenge = random.choice(["LEFT","RIGHT","CLOSER"])
prev_x = None
prev_size = None
done = False

def check_challenge(face):
    global prev_x, prev_size

    x1,y1,x2,y2 = face.bbox.astype(int)
    size = (x2-x1)*(y2-y1)
    nose_x = face.kps[0][0]

    if challenge in ["LEFT","RIGHT"]:
        if prev_x is None:
            prev_x = nose_x
            return False
        diff = nose_x - prev_x
        prev_x = nose_x
        if challenge=="LEFT" and diff < -18:
            return True
        if challenge=="RIGHT" and diff > 18:
            return True

    if challenge=="CLOSER":
        if prev_size is None:
            prev_size = size
            return False
        if size - prev_size > 4000:
            return True

    return False

# ================= AUTO ATTENDANCE =================
last_seen = {}
cooldown = 30   # seconds

def mark_attendance(name, status):
    with open("attendance.csv","a",newline="") as f:
        csv.writer(f).writerow([name,time.ctime(),status])
    print(name,status)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

if not os.path.exists("attendance.csv"):
    with open("attendance.csv","w",newline="") as f:
        csv.writer(f).writerow(["Name","Time","Type"])

print("AUTO ATTENDANCE ACTIVE")

recent = []   # multi-frame verification buffer

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = app.get(frame)
    matched_name = "Unknown"
    best_sim = 0

    if len(faces)>0:
        face = faces[0]
        x1,y1,x2,y2 = face.bbox.astype(int)

        # -------- LIVENESS CHECK --------
        done = check_challenge(face)

        # -------- RECOGNITION --------
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        sims = np.dot(db_embeddings, emb)
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_name = names[best_idx]

        if best_sim > 0.68:
            recent.append(best_name)
        else:
            recent.clear()

        if len(recent) >= 4:
            matched_name = max(set(recent), key=recent.count)
        else:
            matched_name = "Verifying"

        # -------- DISPLAY STATUS --------
        if not done and matched_name not in last_seen:
            status = f"DO: {challenge}"
            color=(0,255,255)
        else:
            status = "REAL"
            color=(0,255,0)

            now = time.time()
            last = last_seen.get(matched_name,0)

            if matched_name!="Unknown" and matched_name!="Verifying" and now-last > cooldown:
                if matched_name not in last_seen:
                    mark_attendance(matched_name,"IN")
                else:
                    mark_attendance(matched_name,"OUT")
                last_seen[matched_name]=now

                challenge=random.choice(["LEFT","RIGHT","CLOSER"])
                prev_x=None; prev_size=None; done=False
                recent.clear()

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,status,(x1,y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        cv2.putText(frame,f"{matched_name} {best_sim:.2f}",
                    (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Auto Attendance", frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
