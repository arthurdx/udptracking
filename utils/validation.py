import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import f1_score
import csv
import os

# ==== CONFIGURAÇÕES ====
MOVEMENT_THRESHOLD = 0.8
DETECTION_COOLDOWN = 60
YOLO_MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 256

def human_detected(frame, model):
    results = model(frame, imgsz=IMG_SIZE, classes=[0])
    return len(results[0].boxes.cls) > 0

def load_annotation_csv(path, start_frame, total_frames):
    reference = np.zeros(total_frames, dtype=int)

    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                absolute_idx = int(row[0])
                coords = list(map(int, row[1:]))
            except ValueError:
                print(f"Pulando linha inválida: {row}")
                continue

            relative_idx = absolute_idx - start_frame
            if 0 <= relative_idx < total_frames and any(coords):
                reference[relative_idx] = 1

    return reference

def summarize_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    summary = []
    prev_gray = None
    cooldown_timer = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            if np.mean(diff) < MOVEMENT_THRESHOLD:
                summary.append(0)
                prev_gray = gray
                frame_idx += 1
                continue

        if cooldown_timer > 0:
            cooldown_timer -= 1
            summary.append(1)
            prev_gray = gray
            frame_idx += 1
            continue

        detected = human_detected(frame, model)
        if detected:
            cooldown_timer = DETECTION_COOLDOWN
            summary.append(1)
        else:
            summary.append(0)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return summary

def main():
    video_path = "videos/conv8A_test1/output.mp4"
    annot_path = "videos/Ground_Truth_Annotations/groundTruth_conv8A_test1.csv"
    start_frame = 8600  

    model = YOLO(YOLO_MODEL_PATH)

    print("Processando vídeo...")
    machine_summary = summarize_video(video_path, model)
    num_frames = len(machine_summary)

    print("Carregando anotação...")
    reference = load_annotation_csv(annot_path, start_frame, num_frames)

    min_len = min(len(machine_summary), len(reference))
    f1 = f1_score(reference[:min_len], machine_summary[:min_len])
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()
