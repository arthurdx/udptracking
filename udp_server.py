import cv2
import socket
import struct

# Server address and port
server_address = ('ip_host', 9999)

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open a video file or capture from a camera
video_capture = cv2.VideoCapture('video.mp4')  # Change to 0 for webcam

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Encode the frame
    encoded, buffer = cv2.imencode('.jpg', frame)
    
    # Send the size of the frame
    size = len(buffer)
    server_socket.sendto(struct.pack('!I', size), server_address)
    
    # Send the frame
    server_socket.sendto(buffer, server_address)

video_capture.release()
server_socket.close()