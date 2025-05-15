import cv2
import socket
import struct
import sys
import threading
from datetime import datetime
import logging
import numpy as np
from ultralytics import YOLO

MOVEMENT_THRESHOLD = 5

logging.basicConfig(
    filename='./logs/server.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

if len(sys.argv) < 2:
    print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
    sys.exit(1)

path = sys.argv[1]
if path == '0':
    path = 0

model = YOLO('yolo11n.pt')

frame_id = 0



# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(("0.0.0.0", 9999))

print("Aguardando conexão do cliente...")
_, client_address = server_socket.recvfrom(16)
print(f"Cliente detectado: {client_address}")


# Open a video file or capture from a camera
video_capture = cv2.VideoCapture(path)  # Change to 0 for webcam
video_capture.set(cv2.CAP_PROP_FPS, 24)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Starting video stream...")

prev_frame = None

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

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        print("Frame difference mean:", np.mean(frame_diff))
        if np.mean(frame_diff) < MOVEMENT_THRESHOLD:
            print("No motion - skipping processing")
            prev_frame = gray_frame.copy()
            continue

    if human_detected(frame):
        encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30]) 
    
        # Send the size of the frameencoded, buffer = cv2.imencode('.jpg', frame)
        size = len(buffer)
        server_socket.sendto(struct.pack('!I d I', frame_id, timestamp, size), client_address)   
        # Send the frame
        server_socket.sendto(buffer.tobytes(), client_address)

        logging.info(
          f'frame_id=#{frame_id}   '
          f'size={size} bytes   '
          f'sent_at={timestamp}   '
        )
    
    prev_frame = gray_frame.copy()


video_capture.release()
server_socket.close()