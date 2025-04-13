import cv2
import socket
import struct

# Server address and port
server_address = ('localhost', 9999)

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open a video file or capture from a camera
video_capture = cv2.VideoCapture(0)  # Change to 0 for webcam
video_capture.set(cv2.CAP_PROP_FPS, 24)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting video stream...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Encode the frame
    encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    
    # Send the size of the frameencoded, buffer = cv2.imencode('.jpg', frame)
    size = len(buffer)
    server_socket.sendto(struct.pack('!I', size), server_address)
    
    # Send the frame
    server_socket.sendto(buffer, server_address)

video_capture.release()
server_socket.close()