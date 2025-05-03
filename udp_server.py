import cv2
import socket
import struct
import sys
import threading
from datetime import datetime
from pythonjsonlogger import jsonlogger
import logging
import numpy as np
from ultralytics import YOLO

MOVEMENT_THRESHOLD = 3.8

log_filename = datetime.now().strftime("./logs/server_%Y%m%d_%H%M%S.ndjson")

logger = logging.getLogger("udp_server")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler(log_filename)
formatter = jsonlogger.JsonFormatter()

log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

if len(sys.argv) < 1:
    print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
    sys.exit(1)

path = sys.argv[1]
if path == '0':
    path = 0

model = YOLO('yolo11n.pt')

frame_id = 0

# Server address and port
server_address = ('localhost', 9999)

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open a video file or capture from a camera
video_capture = cv2.VideoCapture(path)  # Change to 0 for webcam
video_capture.set(cv2.CAP_PROP_FPS, 24)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Starting video stream...")

prev_frame = None
was_processed = None
detected_human = None

def human_detected(frame):
    results = model.track(frame, persist=True, classes=[0])
    return len(results[0].boxes.cls) > 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        logging.error("Falha ao capturar frame")
        break

    frame_id += 1
    timestamp = datetime.now().timestamp()
    
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30]) 
    
    size = len(buffer)

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        print("Frame difference mean:", np.mean(frame_diff))
        if np.mean(frame_diff) < MOVEMENT_THRESHOLD:
            print("No motion - skipping processing")
            prev_frame = gray_frame.copy()
            logger.info("frame_info", extra = {
                "frame_id": frame_id,
                "size_bytes": size,
                "timestamp": timestamp,
                "was_processed": False,
                "was_sent": False
            })
            continue
        else:
            was_processed = True

    if was_processed:
        detected_human = human_detected(frame)
    if detected_human:
        
        server_socket.sendto(struct.pack('!I d I', frame_id, timestamp, size), server_address)   
        # Send the frame
        server_socket.sendto(buffer.tobytes(), server_address)


    
    prev_frame = gray_frame.copy()

    logger.info("frame info", extra={
    "frame_id": frame_id,
    "size_bytes": size,
    "timestamp": timestamp,
    "was_processed": was_processed,
    "was_sent": detected_human,
    })


video_capture.release()
server_socket.close()