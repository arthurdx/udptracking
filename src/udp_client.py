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
import threading
import queue

# Configuração de logging
log_filename = datetime.now().strftime("./logs/client_%Y%m%d_%H%M%S.ndjson")

logger = logging.getLogger("udp_client")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler(log_filename)
formatter = jsonlogger.JsonFormatter()

log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

# Constantes do protocolo
HEADER_SIZE = 20  # frame_id (4) + timestamp (8) + size (4) + sequence (4)
MAX_PACKET_SIZE = 65507  # Tamanho máximo UDP - overhead
BUFFER_SIZE = 65536  # Buffer de recepção maior

class UDPClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        self.disconnect_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = True
        self.expected_sequence = 0
        
    def connect(self):
        """Estabelece conexão com o servidor"""
        try:
            self.client_socket.sendto(b"hello", self.server_address)
            logger.info("Conexão estabelecida com o servidor", extra={
                "server_address": self.server_address
            })
            return True
        except Exception as e:
            logger.error("Erro ao conectar com o servidor", extra={
                "error": str(e),
                "server_address": self.server_address
            })
            return False
    
    def receive_frame_data(self):
        """Recebe dados do frame de forma atômica"""
        try:
            # Recebe o cabeçalho primeiro
            header_data, addr = self.client_socket.recvfrom(HEADER_SIZE)
            if len(header_data) != HEADER_SIZE:
                logger.warning("Cabeçalho incompleto recebido", extra={
                    "received_size": len(header_data),
                    "expected_size": HEADER_SIZE
                })
                return None
                
            # Desempacota o cabeçalho
            frame_id, timestamp, size, sequence = struct.unpack('!I d I I', header_data)
            
            # Verifica se é o pacote esperado
            if sequence != self.expected_sequence:
                logger.warning("Pacote fora de ordem", extra={
                    "expected_sequence": self.expected_sequence,
                    "received_sequence": sequence,
                    "frame_id": frame_id
                })
                # Pode implementar retransmissão aqui se necessário
                return None
            
            # Recebe o frame em chunks se necessário
            frame_data = b""
            remaining_size = size
            
            while remaining_size > 0:
                chunk_size = min(remaining_size, MAX_PACKET_SIZE)
                chunk, addr = self.client_socket.recvfrom(chunk_size)
                frame_data += chunk
                remaining_size -= len(chunk)
            
            # Incrementa o número de sequência esperado
            self.expected_sequence += 1
            
            return {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'size': size,
                'sequence': sequence,
                'frame_data': frame_data
            }
            
        except Exception as e:
            logger.error("Erro ao receber dados do frame", extra={
                "error": str(e)
            })
            return None
    
    def process_frame(self, frame_data):
        """Processa o frame recebido"""
        try:
            frame = cv2.imdecode(np.frombuffer(frame_data['frame_data'], dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("Falha ao decodificar frame", extra={
                    "frame_id": frame_data['frame_id']
                })
                return None
            
            recv_time = datetime.now()
            send_time = datetime.fromtimestamp(frame_data['timestamp'])
            latency = (recv_time - send_time).total_seconds() * 1000
            
            logger.info("frame_info", extra={
                "frame_id": frame_data['frame_id'],
                "timestampMs": frame_data['timestamp'],
                "recv_timeMs": recv_time.timestamp(),
                "send_timeMs": send_time.timestamp(),
                "latencyMs": latency,
                "size": frame_data['size'],
                "sequence": frame_data['sequence']
            })
            
            return frame
            
        except Exception as e:
            logger.error("Erro ao processar frame", extra={
                "error": str(e),
                "frame_id": frame_data.get('frame_id', 'unknown')
            })
            return None
    
    def run(self):
        """Loop principal do cliente"""
        if not self.connect():
            return
        
        try:
            while self.running:
                frame_data = self.receive_frame_data()
                if frame_data is None:
                    continue
                
                frame = self.process_frame(frame_data)
                if frame is None:
                    continue
                
                cv2.imshow('Video', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupção do usuário detectada")
        except Exception as e:
            logger.error("Erro no loop principal", extra={"error": str(e)})
        finally:
            self.cleanup()
    
    def disconnect(self):
        """Solicita desconexão e recebe logs do servidor"""
        try:
            time.sleep(1.0)
            self.disconnect_socket.sendto(b"bye", (self.server_address[0], self.server_address[1] - 1))
            logger.info("Solicitando log do servidor...")
            
            log_data = b""
            while True:
                size_data, _ = self.disconnect_socket.recvfrom(4)
                chunk_size = struct.unpack("!I", size_data)[0]
                if chunk_size == 0:
                    break
                chunk, _ = self.disconnect_socket.recvfrom(chunk_size)
                log_data += chunk
            
            os.makedirs("./logs", exist_ok=True)
            with open(log_filename.replace("client", "server"), "wb") as f:
                f.write(log_data)
            
            logger.info("Log do servidor salvo com sucesso")
            
        except Exception as e:
            logger.error("Erro ao receber logs do servidor", extra={"error": str(e)})
    
    def cleanup(self):
        """Limpa recursos"""
        self.running = False
        self.client_socket.close()
        self.disconnect_socket.close()
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 3:
        print("Usage: python udp_client.py <ip> <port>")
        sys.exit(1)
    
    ip = sys.argv[1]
    port = int(sys.argv[2])
    server_address = (ip, port)
    
    client = UDPClient(server_address)
    
    try:
        client.run()
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
