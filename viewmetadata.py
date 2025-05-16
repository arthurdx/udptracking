import cv2
import subprocess
import json
import os
import sys

def extrair_metadados_ffprobe(caminho_video):
    comando = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", caminho_video
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    metadados = json.loads(resultado.stdout)

    print(json.dumps(metadados, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python metadados_video.py <caminho_do_video>")
    else:
        extrair_metadados_ffprobe(sys.argv[1])