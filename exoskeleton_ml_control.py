# exoskeleton_ml_control.py
# Machine learning control system for wearable exoskeleton using EMG and IMU data

import numpy as np
import pandas as pd
import joblib
import serial
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# --- Load Pretrained ML Model ---
emg_model = load_model('emg_intent_model.h5')
scaler = joblib.load('scaler.save')

# --- Setup Serial Communication with Microcontroller ---
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

# --- Constants ---
EMG_CHANNELS = 8
IMU_FEATURES = 6  # accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
WINDOW_SIZE = 50

# --- Rolling Window for Time-Series ---
emg_buffer = np.zeros((WINDOW_SIZE, EMG_CHANNELS))
imu_buffer = np.zeros((WINDOW_SIZE, IMU_FEATURES))

# --- Parse Serial Data from EMG + IMU Sensors ---
def parse_serial_data(line):
    try:
        data = list(map(float, line.decode('utf-8').strip().split(',')))
        return data
    except:
        return None

# --- Prediction and Control Loop ---
def control_loop():
    while True:
        if ser.in_waiting:
            line = ser.readline()
            data = parse_serial_data(line)
            if data and len(data) == (EMG_CHANNELS + IMU_FEATURES):
                emg = np.array(data[:EMG_CHANNELS])
                imu = np.array(data[EMG_CHANNELS:])

                # Update rolling buffers
                global emg_buffer, imu_buffer
                emg_buffer = np.roll(emg_buffer, -1, axis=0)
                imu_buffer = np.roll(imu_buffer, -1, axis=0)
                emg_buffer[-1] = emg
                imu_buffer[-1] = imu

                # Feature vector
                if np.all(emg_buffer) and np.all(imu_buffer):
                    features = np.hstack([emg_buffer.flatten(), imu_buffer.flatten()])
                    features = scaler.transform([features])
                    prediction = emg_model.predict(features)
                    movement_class = np.argmax(prediction)

                    # Send command to motor controller
                    send_motor_command(movement_class)

# --- Translate Movement Class to Motor Commands ---
def send_motor_command(class_idx):
    # 0: rest, 1: lift, 2: extend, 3: bend
    command_map = ['REST', 'LIFT', 'EXTEND', 'BEND']
    command = command_map[class_idx]
    ser.write(f'{command}\n'.encode())
    print(f"Predicted: {command}")

# --- Run ---
if __name__ == '__main__':
    try:
        control_loop()
    except KeyboardInterrupt:
        print("Shutting down...")
        ser.close()
