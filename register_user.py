import os
os.environ["INSIGHTFACE_HOME"] = r"E:\attendance_ai\insightface_models"

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from insightface.app import FaceAnalysis

DB_PATH = "db/embeddings.npz"

# ---------- Load DB ----------
if os.path.exists(DB_PATH):
    data = np.load(DB_PATH, allow_pickle=True)
    names = list(data["names"])
    embeddings = list(data["embeddings"])
else:
    names = []
    embeddings = []

# ---------- InsightFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640,640))

# ---------- Register Logic ----------
def start_register():
    username = simpledialog.askstring("Register", "Enter Student Name:")
    if not username:
        return

    cap = cv2.VideoCapture(0)
    samples = []

    messagebox.showinfo("Info",
        "Look at camera\nTurn head LEFT and RIGHT\nCollecting 20 samples")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)

        if len(faces)>0:
            face = faces[0]
            x1,y1,x2,y2 = face.bbox.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            samples.append(face.embedding)

            cv2.putText(frame,f"Samples: {len(samples)}/20",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
        if len(samples)>=20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples)<10:
        messagebox.showerror("Fail","Not enough samples")
        return

    mean_emb = np.mean(samples, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)  # normalization

    names.append(username)
    embeddings.append(mean_emb)

    os.makedirs("db", exist_ok=True)
    np.savez(DB_PATH, names=np.array(names), embeddings=np.array(embeddings))

    messagebox.showinfo("Success", f"{username} registered successfully")

# ---------- GUI ----------
root = tk.Tk()
root.title("Register Student")
root.geometry("300x200")

tk.Label(root, text="Student Registration",
         font=("Arial",14,"bold")).pack(pady=20)

tk.Button(root, text="Register New Student",
          width=25, height=2,
          command=start_register).pack(pady=20)

tk.Button(root, text="Exit", width=25,
          command=root.quit).pack()

root.mainloop()
