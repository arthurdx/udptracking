import cv2
import socket
import struct
import numpy as np
import logging
from pythonjsonlogger import jsonlogger
from datetime import datetime

log_filename = datetime.now().strftime("./logs/client_%Y%m%d_%H%M%S.ndjson")

logger = logging.getLogger("udp_client")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler(log_filename)
formatter = jsonlogger.JsonFormatter()

log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

# Server address and port
server_address = ('localhost', 9999)

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind(server_address)

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
    }

    )

    # out.write(frame)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()

# out.release()

cv2.destroyAllWindows()