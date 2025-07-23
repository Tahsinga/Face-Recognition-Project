from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading
import queue
import torch
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time
prev_time = time.time()

# Use GPU if available
# TO RUN THE VN: venv\Scripts\activate
# RECREATE THE ENVIRONMENT: pip install -r requirements.txt

device = "cpu" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# Load known faces using InsightFace
known_embeddings, known_names = [], []
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # CPU
face_app.prepare(ctx_id=0)

base_dir = r"D:\TASHINGA_PROJECTS\OBJECT_DETECTION\Face detection system\train_images"
print("📂 Loading known faces (GPU)…")

for name in os.listdir(base_dir):
    pdir = os.path.join(base_dir, name)
    if not os.path.isdir(pdir):
        continue
    for file in os.listdir(pdir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(pdir, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_app.get(img)
            if faces:
                known_embeddings.append(faces[0].normed_embedding)
                known_names.append(name)
            else:
                print(f"⚠️ No face found in {file}")

print(f"✅ Loaded {len(known_embeddings)} known faces.")

# Thread-safe variables
detection_queue = queue.Queue(maxsize=1)
recognition_queue = queue.Queue(maxsize=1)
face_boxes, face_names = [], []
lock = threading.Lock()

def yolo_worker():
    while True:
        frame = detection_queue.get()
        small = cv2.resize(frame, (224, 224))  # faster
        result = model(small, verbose=False)[0]
        h, w = frame.shape[:2]
        dets = []

        for box in result.boxes:
            if float(box.conf[0]) < 0.6:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            scale_x, scale_y = w / 224, h / 224

            # Reduce the green box size (tight fit or 3% padding)
            pad_x = int((x2 - x1) * 0.03)
            pad_y = int((y2 - y1) * 0.03)

            x1 = max(int(x1 * scale_x) - pad_x, 0)
            y1 = max(int(y1 * scale_y) - pad_y, 0)
            x2 = min(int(x2 * scale_x) + pad_x, w)
            y2 = min(int(y2 * scale_y) + pad_y, h)

            dets.append((x1, y1, x2, y2))

        if recognition_queue.full():
            recognition_queue.get_nowait()
        recognition_queue.put((frame, dets))
        detection_queue.task_done()

def recognition_worker():
    global face_boxes, face_names

    while True:
        frame, detections = recognition_queue.get()
        names, boxes = [], []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if hasattr(recognition_worker, "skip") and recognition_worker.skip % 2 != 0:
            recognition_worker.skip += 1
            recognition_queue.task_done()
            continue

        recognition_worker.skip = getattr(recognition_worker, "skip", 0) + 1

        for (x1, y1, x2, y2) in detections:
            face_region = rgb[y1:y2, x1:x2]
            if face_region.size == 0 or face_region.shape[0] < 30 or face_region.shape[1] < 30:
                continue

            faces = face_app.get(face_region)
            if not faces:
                continue

            face = faces[0]
            fx1, fy1, fx2, fy2 = [int(coord) for coord in face.bbox]
            gx1 = x1 + fx1
            gy1 = y1 + fy1
            gx2 = x1 + fx2
            gy2 = y1 + fy2

            emb = face.normed_embedding.reshape(1, -1)
            sims = cosine_similarity(known_embeddings, emb).flatten()
            idx = np.argmax(sims)
            name = known_names[idx] if sims[idx] > 0.45 else "Unknown"

            names.append(name)
            boxes.append((gx1, gy1, gx2, gy2))

        with lock:
            face_boxes = boxes
            face_names = names
        recognition_queue.task_done()

# Start worker threads
threading.Thread(target=yolo_worker, daemon=True).start()
threading.Thread(target=recognition_worker, daemon=True).start()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Starting real-time face recognition… Press 'q' to quit.")
frame_skip = 0.5  # Reduce skip to improve speed
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inside the while loop, before imshow
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    frame_count += 1
    if frame_count % frame_skip == 0:
        if not detection_queue.full():
            detection_queue.put(frame.copy())

    with lock:
        for (x1, y1, x2, y2), name in zip(face_boxes, face_names):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("🚀 Real-Time Face Recognition (GPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
