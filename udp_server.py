import cv2
import socket
import struct
import sys
import threading
import numpy as np
from ultralytics import YOLO

MOVEMENT_THRESHOLD = 5

if len(sys.argv) < 1:
    print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
    sys.exit(1)

path = sys.argv[1]
if path == '0':
    path = 0

model = YOLO('yolo11n.pt')

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

def human_detected(frame):
    results = model.track(frame, persist=True, classes=[0])
    return len(results[0].boxes.cls) > 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

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
        server_socket.sendto(struct.pack('!I', size), server_address)   
        # Send the frame
        server_socket.sendto(buffer.tobytes(), server_address)
    
    prev_frame = gray_frame.copy()


video_capture.release()
server_socket.close()