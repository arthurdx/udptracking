import cv2
import socket
import struct
import numpy as np
import datetime

# Server address and port
server_address = ('localhost', 9999)

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind(server_address)
videoname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(videoname)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ou 'MJPG', 'MP4V' etc.
out = cv2.VideoWriter(f'./videos/{videoname}.avi', fourcc, 24.0, (640, 360))
if not out.isOpened():
    print("Erro ao abrir VideoWriter")
    client_socket.close()
    exit(1)

while True:
    # Receive the size of the frame
    size_data, _ = client_socket.recvfrom(4)
    size = struct.unpack('!I', size_data)[0]
    
    # Receive the frame
    buffer, _ = client_socket.recvfrom(size)
    
    # Decode the frame
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # Display the frame
    out.write(frame)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
out.release()
cv2.destroyAllWindows()