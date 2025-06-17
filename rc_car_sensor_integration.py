# rc_car_sensor_integration.py
# Simulated sensor fusion for self-driving RC car using Raspberry Pi and Arduino

import serial
import time
import threading
import smbus  # I2C communication for IMU
import cv2     # Camera input
import numpy as np

# --- Setup Serial Communication with Arduino ---
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Wait for connection to establish

# --- Setup I2C for IMU (e.g., MPU6050) ---
bus = smbus.SMBus(1)
IMU_ADDR = 0x68
bus.write_byte_data(IMU_ADDR, 0x6B, 0)  # Wake up the IMU

def read_imu():
    accel_x = read_word_2c(IMU_ADDR, 0x3B) / 16384.0
    gyro_z = read_word_2c(IMU_ADDR, 0x47) / 131.0
    return accel_x, gyro_z

def read_word_2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg+1)
    val = (high << 8) + low
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val

# --- Camera Thread ---
def camera_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale and detect lane (placeholder logic)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Placeholder: determine direction based on simple pixel threshold
        left = np.sum(edges[:, :int(edges.shape[1]/2)])
        right = np.sum(edges[:, int(edges.shape[1]/2):])
        if left > right:
            send_command('L')  # Turn Left
        elif right > left:
            send_command('R')  # Turn Right
        else:
            send_command('F')  # Go Forward

        time.sleep(0.1)

def send_command(command):
    arduino.write(f'{command}\n'.encode())
    print(f'Sent to Arduino: {command}')

# --- IMU Thread ---
def imu_loop():
    while True:
        accel_x, gyro_z = read_imu()
        print(f"Accel X: {accel_x:.2f}, Gyro Z: {gyro_z:.2f}")
        time.sleep(0.2)

# --- Main ---
if __name__ == '__main__':
    t1 = threading.Thread(target=camera_loop)
    t2 = threading.Thread(target=imu_loop)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
