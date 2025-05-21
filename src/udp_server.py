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
import psutil


MOVEMENT_THRESHOLD = 5
DETECT_EVERY_N_FRAMES = 2
DETECTION_COOLDOWN = 48

mask = DETECT_EVERY_N_FRAMES - 1
cooldown_timer = 0

log_filename = datetime.now().strftime("./logs/server_%Y%m%d_%H%M%S.ndjson")

logger = logging.getLogger("udp_server")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler(log_filename)
formatter = jsonlogger.JsonFormatter()

log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

process = psutil.Process()


if len(sys.argv) < 2:
    print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
    sys.exit(1)

path = sys.argv[1]
if path == '0':
    path = 0

model = YOLO('yolo11n.pt')

frame_id = 0



server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(("0.0.0.0", 9999))

print("Aguardando conexão do cliente...")
_, client_address = server_socket.recvfrom(16)
print(f"Cliente detectado: {client_address}")


video_capture = cv2.VideoCapture(path)  
video_capture.set(cv2.CAP_PROP_FPS, 24)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

print("Starting video stream...")

prev_frame = None
was_processed = None
detected_human = None

def human_detected(frame):
    results = model(frame, imgsz=256, classes=[0])
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

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        print("Frame difference mean:", np.mean(frame_diff))
        if np.mean(frame_diff) < MOVEMENT_THRESHOLD:
            print("No motion - skipping processing")
            prev_frame = gray_frame.copy()
            logger.info("sem movimento", extra={
                "frameId": frame_id,
                "timestamp": timestamp,
                "processed": False,
                "sent": False
            })
            prev_frame = gray_frame.copy()
            continue

    detected = human_detected(frame)

    if detected:
        cooldown_timer = DETECTION_COOLDOWN

    cpu = process.cpu_percent(interval=None)           
    mem = process.memory_info().rss                    
    mem_pct = process.memory_percent() 

    if detected or cooldown_timer > 0:
        if cooldown_timer > 0:
            cooldown_timer -= 1
        encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30]) 
    
     
        size = len(buffer)

        server_socket.sendto(struct.pack('!I d I', frame_id, timestamp, size), client_address)   
 
        server_socket.sendto(buffer.tobytes(), client_address)

        

        logger.info("", extra={
            "frame_id": frame_id,
            "timestamp": timestamp,
            "processed": True,
            "sent": True,
            "size_bytes": size,
            "cpu_percent": cpu,
            "mem_rss_bytes": mem,
            "mem_percent": mem_pct
        })
    else:
        logger.info("", extra={
            "frame_id": frame_id,
            "timestamp": timestamp,
            "processed": True,
            "sent": False,
            "cpu_percent": cpu,
            "mem_rss_bytes": mem,
            "mem_percent": mem_pct
        })


    prev_frame = gray_frame.copy()
    
    

video_capture.release()
server_socket.close()