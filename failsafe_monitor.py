# failsafe_monitor.py
# Background failsafe service for emergency stop if sensors go offline or no signal is received

import time
from communication_handler import SerialHandler

serial = SerialHandler('/dev/ttyUSB0', 9600)
last_ping_time = time.time()
PING_TIMEOUT = 3  # seconds

print("Failsafe monitor active...")

try:
    while True:
        line = serial.read_line()
        if line == 'PING':
            last_ping_time = time.time()

        # If no ping received in time, stop the car
        if time.time() - last_ping_time > PING_TIMEOUT:
            print("[Failsafe] Lost signal. Sending STOP.")
            serial.send('S')
            last_ping_time = time.time()  # avoid spamming

        time.sleep(0.2)

except KeyboardInterrupt:
    serial.send('S')
    serial.close()
    print("Failsafe monitor stopped.")
