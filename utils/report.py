import sys
import os
import json

def read_log_file(filename: str) -> list:
    log_dict_list = []
    with open(filename, 'r') as file:
        current_line = 0
        for line in file:
            line_dict = json.loads(line)
            log_dict_list.append(line_dict)
    return log_dict_list

def main():
    lost_frames = []
    total_byte_size = 0
    if len(sys.argv) < 2:
        print("Usage: python scriptname.py path_to_logs")
        return
    print("run")
    path_to_logs = sys.argv[1]
    for filename in os.listdir(path_to_logs):
        if filename.startswith('client'):
            client_log = read_log_file(path_to_logs + filename)
        else:
            server_log = read_log_file(path_to_logs + filename)
    for server_frame in server_log:
        if server_frame["was_sent"]:
            frame_id = server_frame['frame_id']
            total_byte_size += server_frame['size_bytes']
            found = any(frame['frame_id'] == frame_id for frame in client_log)
            if not found:
                lost_frames.append(frame_id)
    print(lost_frames)
    loss = (len(lost_frames) / len(client_log)) * 100
    flow = (total_byte_size) / ((server_log[-1]['timestamp'] - server_log[1]['timestamp']) * 1000)
    latency = sum(line["latencyMs"] for line in client_log) / len(client_log)
    print(f"Total de patores perdidos {len(lost_frames)}\
    \nHouve uma perda de {loss:.3f}% dos pacotes\
    \na vazão da rede foi de {flow:.3f}MBps\
    \na latência média foi de {latency:.3f}ms")

    
main()