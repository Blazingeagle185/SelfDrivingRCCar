# test_hardware.py
# Diagnostics tool to verify hardware components before full system run

import cv2
import smbus
import time
from communication_handler import SerialHandler

# Test camera
print("Testing camera...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Camera OK")
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
else:
    print("Camera ERROR")

# Test IMU
print("Testing IMU...")
bus = smbus.SMBus(1)
try:
    bus.write_byte_data(0x68, 0x6B, 0)
    val = bus.read_byte_data(0x68, 0x75)
    print(f"IMU OK, WHO_AM_I register: {val}")
except Exception as e:
    print(f"IMU ERROR: {e}")

# Test Serial to Arduino
print("Testing Arduino serial communication...")
serial = SerialHandler()
serial.send("S")  # send stop command
line = serial.read_line()
serial.close()
if line:
    print("Serial communication OK")
else:
    print("No response from Arduino")
