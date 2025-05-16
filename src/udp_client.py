import cv2
import socket
import struct
import numpy as np
import logging
from pythonjsonlogger import jsonlogger
import sys
from datetime import datetime

logger = logging.getLogger()

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

if len(sys.argv) < 3:
    print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
    sys.exit(1)

ip = sys.argv[1]
port = int(sys.argv[2])

# Server address and port
server_address = (ip, int(port))

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto(b"hello", (ip, int(port)))

# salvar o vídeo
# videoname = datetime.now().strftime("%Y%m%d_%H%M%S")
# print(videoname)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ou 'MJPG', 'MP4V' etc.
# out = cv2.VideoWriter(f'./videos/{videoname}.avi', fourcc, 24.0, (640, 360))
# if not out.isOpened():
#     print("Erro ao abrir VideoWriter")
#     client_socket.close()
#     exit(1)

while True:
    # Receive the size of the frame
    size_data, _ = client_socket.recvfrom(16)

    frame_id, timestamp, size = struct.unpack('!I d I', size_data)

    buffer, _ = client_socket.recvfrom(size)
    
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        continue

    recv_time = datetime.now()
    send_time = datetime.fromtimestamp(timestamp)
    latency = (recv_time - send_time).total_seconds() * 1000  # em milissegundos
    
    logging.info(
          f'frame_id={frame_id}  '
          f'from_server=ip:{server_address[0]}port:{server_address[1]}  '
          f'size={size} bytes  '
          f'sent_at={send_time.time()}  '
          f'latency={latency:.4f} ms')

    # out.write(frame)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()

# out.release()

cv2.destroyAllWindows()