import cv2
import socket
import struct
import threading
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Server address and port
server_address = ('localhost', 9999)

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open a video file or capture from a camera
video_capture = cv2.VideoCapture(0)  # Change to 0 for webcam
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

    if human_detected(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            if np.mean(frame_diff) < 10:
                print("No significant motion detected.")
                continue
        prev_frame = gray_frame.copy()
                
        encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50]) 
    
        # Send the size of the frameencoded, buffer = cv2.imencode('.jpg', frame)
        size = len(buffer)
        server_socket.sendto(struct.pack('!I', size), server_address)   
        # Send the frame
        server_socket.sendto(buffer.tobytes(), server_address)


video_capture.release()
server_socket.close()