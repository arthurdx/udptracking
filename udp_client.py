import cv2
import socket
import struct
import numpy as np

# Server address and port
server_address = ('ip_host', 9999)

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind(server_address)

while True:
    # Receive the size of the frame
    size_data, _ = client_socket.recvfrom(4)
    size = struct.unpack('!I', size_data)[0]
    
    # Receive the frame
    buffer, _ = client_socket.recvfrom(size)
    
    # Decode the frame
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()