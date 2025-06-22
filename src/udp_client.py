import cv2
import socket
import struct
import numpy as np
import logging
from pythonjsonlogger import jsonlogger
from datetime import datetime
import sys
import time
import os

log_filename = datetime.now().strftime("./logs/client_%Y%m%d_%H%M%S.ndjson")

logger = logging.getLogger("udp_client")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler(log_filename)
formatter = jsonlogger.JsonFormatter()

log_handler.setFormatter(formatter)
logger.addHandler(log_handler)


if len(sys.argv) < 3:
    print("Usage: python udp_client.py <ip> <port>")
    sys.exit(1)

ip = sys.argv[1]
port = sys.argv[2]
# Server address and port
server_address = (ip, int(port))

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto(b"hello", server_address)

disconnect_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    
    logger.info("frame_info", extra = {
        "frame_id": frame_id,
        "timestampMs": timestamp,
        "recv_timeMs": recv_time.timestamp(),
        "send_timeMs": send_time.timestamp(),
        "latencyMs": latency,
        "size": size,
    })

    # out.write(frame)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(1.0)
client_socket.close() 

disconnect_socket.sendto(b"bye", (ip, int(port)- 1))
print("Solicitando log do servidor...")

log_data = b""
while True:
    size_data, _ = disconnect_socket.recvfrom(4)
    chunk_size = struct.unpack("!I", size_data)[0]
    if chunk_size == 0:
        break
    chunk, _ = disconnect_socket.recvfrom(chunk_size)
    log_data += chunk  

os.makedirs("./logs", exist_ok=True)
with open(log_filename.replace("client", "server"), "wb") as f:
    f.write(log_data)

print("Log do servidor salvo.")
disconnect_socket.close()
# out.release()

cv2.destroyAllWindows()
