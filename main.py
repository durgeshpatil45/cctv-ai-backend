from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import cv2
from ultralytics import YOLO

app = FastAPI()
@app.get("/health")
def health():
    return {"status": "ok"}


model = YOLO("yolov8n.pt")

VIDEOS_DIR = "videos"
RESULTS_DIR = "results"

import os
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    path = f"{VIDEOS_DIR}/{video_id}.mp4"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"video_id": video_id}


@app.post("/start-analysis/{video_id}")
def analyze_video(video_id: str):
    cap = cv2.VideoCapture(f"{VIDEOS_DIR}/{video_id}.mp4")
    results = []
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        for box in detections.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                results.append({
                    "frame": frame_no,
                    "bbox": [x1, y1, x2, y2]
                })

        frame_no += 1

    cap.release()

    import json
    with open(f"{RESULTS_DIR}/{video_id}.json", "w") as f:
        json.dump(results, f)

    return {"status": "completed"}


@app.get("/results/{video_id}")
def get_results(video_id: str):
    import json
    with open(f"{RESULTS_DIR}/{video_id}.json") as f:
        return json.load(f)

