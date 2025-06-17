# manual_control_interface.py
# Optional script to manually control RC car using keyboard

import keyboard
from communication_handler import SerialHandler

serial = SerialHandler('/dev/ttyUSB0', 9600)

print("Press W/A/S/D for motion, Q to quit")

try:
    while True:
        if keyboard.is_pressed('w'):
            serial.send('F')
        elif keyboard.is_pressed('s'):
            serial.send('B')
        elif keyboard.is_pressed('a'):
            serial.send('L')
        elif keyboard.is_pressed('d'):
            serial.send('R')
        elif keyboard.is_pressed('q'):
            break

except KeyboardInterrupt:
    pass

serial.send('S')
serial.close()
print("Manual control stopped.")
