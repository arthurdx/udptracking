import cv2
import socket
import struct
import sys
import threading
from datetime import datetime
from pythonjsonlogger import jsonlogger
import logging
import numpy as np
from ultralytics import YOLO
import psutil
import time
import os

# Constantes do protocolo
HEADER_SIZE = 20  # frame_id (4) + timestamp (8) + size (4) + sequence (4)
MAX_PACKET_SIZE = 65507  # Tamanho máximo UDP - overhead
BUFFER_SIZE = 65536  # Buffer de envio maior

class UDPServer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        self.client_address = None
        self.send_video = True
        self.sequence_number = 0
        self.frame_id = 0
        
        # Configuração de logging
        self.log_filename = datetime.now().strftime("./logs/server_%Y%m%d_%H%M%S.ndjson")
        self.setup_logging()
        
        # Configuração do modelo YOLO
        self.model = YOLO('yolo11n.pt')
        
        # Configuração de detecção
        self.MOVEMENT_THRESHOLD = 1.5
        self.DETECT_EVERY_N_FRAMES = 2
        self.DETECTION_COOLDOWN = 60
        self.mask = self.DETECT_EVERY_N_FRAMES - 1
        self.cooldown_timer = 0
        self.prev_frame = None
        
        # Configuração de monitoramento
        self.process = psutil.Process()
        
    def setup_logging(self):
        """Configura o sistema de logging"""
        self.logger = logging.getLogger("udp_server")
        self.logger.setLevel(logging.INFO)
        
        log_handler = logging.FileHandler(self.log_filename)
        formatter = jsonlogger.JsonFormatter()
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)
    
    def human_detected(self, frame):
        """Detecta humanos no frame usando YOLO"""
        try:
            results = self.model(frame, imgsz=256, classes=[0])
            return len(results[0].boxes.cls) > 0
        except Exception as e:
            self.logger.error("Erro na detecção de humanos", extra={"error": str(e)})
            return False
    
    def send_frame_atomic(self, frame, frame_id, timestamp):
        """Envia frame de forma atômica com sequenciamento"""
        try:
            # Codifica o frame
            encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
            frame_data = buffer.tobytes()
            size = len(frame_data)
            
            # Cria o cabeçalho com sequenciamento
            header = struct.pack('!I d I I', frame_id, timestamp, size, self.sequence_number)
            
            # Envia o cabeçalho primeiro
            self.server_socket.sendto(header, self.client_address)
            
            # Envia o frame em chunks se necessário
            remaining_data = frame_data
            while remaining_data:
                chunk_size = min(len(remaining_data), MAX_PACKET_SIZE)
                chunk = remaining_data[:chunk_size]
                self.server_socket.sendto(chunk, self.client_address)
                remaining_data = remaining_data[chunk_size:]
            
            # Incrementa o número de sequência
            self.sequence_number += 1
            
            return size
            
        except Exception as e:
            self.logger.error("Erro ao enviar frame", extra={
                "error": str(e),
                "frame_id": frame_id
            })
            return 0
    
    def listen_for_disconnect(self):
        """Thread para escutar solicitações de desconexão"""
        while self.send_video:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                if data == b"bye":
                    self.logger.info(f"Cliente {addr} solicitou desconexão. Enviando log...")
                    self.send_video = False
                    
                    time.sleep(1.0)
                    
                    # Envia o arquivo de log
                    try:
                        with open(self.log_filename, "rb") as f:
                            while True:
                                chunk = f.read(1024)
                                if not chunk:
                                    break
                                size_chunk = struct.pack("!I", len(chunk))
                                self.server_socket.sendto(size_chunk, addr)
                                self.server_socket.sendto(chunk, addr)
                        
                        # Sinaliza fim do arquivo
                        self.server_socket.sendto(struct.pack("!I", 0), addr)
                        self.logger.info("Log enviado. Encerrando servidor.")
                        
                    except Exception as e:
                        self.logger.error("Erro ao enviar log", extra={"error": str(e)})
                    
                    os._exit(0)
                    
            except Exception as e:
                if self.send_video:  # Só loga se ainda estiver rodando
                    self.logger.error("Erro na thread de escuta", extra={"error": str(e)})
    
    def process_frame(self, frame):
        """Processa um frame individual"""
        self.frame_id += 1
        timestamp = datetime.now().timestamp()
        
        # Aplica flip horizontal
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Verifica movimento
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
            motion_mean = np.mean(frame_diff)
            
            if motion_mean < self.MOVEMENT_THRESHOLD:
                self.logger.info("sem movimento", extra={
                    "frameId": self.frame_id,
                    "timestamp": timestamp,
                    "processed": False,
                    "sent": False,
                    "motion_mean": motion_mean
                })
                self.prev_frame = gray_frame.copy()
                return False
        
        # Verifica cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            size = self.send_frame_atomic(frame, self.frame_id, timestamp)
            self.logger.info("'recarregando' a função de processamento", extra={
                "frame_id": self.frame_id,
                "timestamp": timestamp,
                "size_bytes": size,
                "processed": False,
                "sent": True,
                "cooldown_timer": self.cooldown_timer
            })
            return True
        
        # Detecta humanos
        detected = self.human_detected(frame)
        
        if detected:
            self.cooldown_timer = self.DETECTION_COOLDOWN
        
        # Monitoramento de recursos
        cpu = self.process.cpu_percent(interval=None)
        mem = self.process.memory_info().rss
        mem_pct = self.process.memory_percent()
        
        if detected:
            size = self.send_frame_atomic(frame, self.frame_id, timestamp)
            self.logger.info("humano detectado", extra={
                "frame_id": self.frame_id,
                "timestamp": timestamp,
                "processed": True,
                "sent": True,
                "size_bytes": size,
                "cpu_percent": cpu,
                "mem_rss_bytes": mem,
                "mem_percent": mem_pct,
                "sequence": self.sequence_number - 1
            })
        else:
            self.logger.info("sem detecção", extra={
                "frame_id": self.frame_id,
                "timestamp": timestamp,
                "processed": True,
                "sent": False,
                "cpu_percent": cpu,
                "mem_rss_bytes": mem,
                "mem_percent": mem_pct
            })
        
        self.prev_frame = gray_frame.copy()
        return detected
    
    def run(self):
        """Loop principal do servidor"""
        # Configura o socket
        self.server_socket.bind(("0.0.0.0", 9999))
        self.logger.info("Servidor iniciado na porta 9999")
        
        # Aguarda conexão do cliente
        print("Aguardando conexão do cliente...")
        _, self.client_address = self.server_socket.recvfrom(16)
        self.logger.info(f"Cliente detectado: {self.client_address}")
        
        # Inicia thread de desconexão
        disconnect_thread = threading.Thread(target=self.listen_for_disconnect, daemon=True)
        disconnect_thread.start()
        
        # Configura captura de vídeo
        video_capture = cv2.VideoCapture(self.video_path)
        video_capture.set(cv2.CAP_PROP_FPS, 24)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        
        self.logger.info("Stream de vídeo iniciado")
        
        try:
            while self.send_video:
                ret, frame = video_capture.read()
                if not ret:
                    self.logger.error("Falha ao capturar frame")
                    break
                
                self.process_frame(frame)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupção do usuário detectada")
        except Exception as e:
            self.logger.error("Erro no loop principal", extra={"error": str(e)})
        finally:
            video_capture.release()
            self.server_socket.close()
            self.logger.info("Servidor encerrado")

def main():
    if len(sys.argv) < 2:
        print("Usage: python udp_server.py <video_file_path> or 0 for webcam")
        sys.exit(1)
    
    path = sys.argv[1]
    if path == '0':
        path = 0
    
    server = UDPServer(path)
    server.run()

if __name__ == "__main__":
    main()
