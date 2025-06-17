# communication_handler.py
# Shared utilities for serial communication with Arduino

import serial
import time

class SerialHandler:
    def __init__(self, port='/dev/ttyUSB0', baud=9600, timeout=1):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # allow time for the serial connection to settle

    def send(self, command):
        if not command.endswith('\n'):
            command += '\n'
        self.ser.write(command.encode())
        print(f"[Serial] Sent: {command.strip()}")

    def read_line(self):
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            print(f"[Serial] Received: {line}")
            return line
        return None

    def close(self):
        self.ser.close()
