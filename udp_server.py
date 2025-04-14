import cv2
import socket
import struct
import threading
import tensorflow as tf
import numpy as np
from collections import deque

# base_model = tf.keras.applications.InceptionV3(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax'
# )

# converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
# model = converter.convert()
# with open('inceptionv3.tflite', 'wb') as f:
#     f.write(model)

interpreter = tf.lite.Interpreter(model_path='inceptionv3.tflite')
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Server address and port
server_address = ('localhost', 9999)


# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Open a video file or capture from a camera
video_capture = cv2.VideoCapture(0)  # Change to 0 for webcam
video_capture.set(cv2.CAP_PROP_FPS, 24)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_frame = None
human_threshold = 0.7  
frame_buffer = deque(maxlen=5)

def human_detected(frame):
    resized_frame = cv2.resize(frame, (299, 299))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0][0] > human_threshold

print("Starting video stream...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if human_detected(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            if np.mean(frame_diff) < 20:
                continue
        prev_frame = gray_frame.copy()

        if len(frame_buffer) == frame_buffer.maxlen:
            for f in frame_buffer:
                # Encode the frame
                
                encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    
                # Send the size of the frameencoded, buffer = cv2.imencode('.jpg', frame)
                size = len(buffer)
                server_socket.sendto(struct.pack('!I', size), server_address)   
                # Send the frame
                server_socket.sendto(buffer, server_address)
        frame_buffer.clear()

    



video_capture.release()
server_socket.close()