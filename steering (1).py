import RPi.GPIO as GPIO
from time import sleep

# Motor 1 (Left)
in1 = 24
in2 = 23
en1 = 25

# Motor 2 (Right)
in3 = 17
in4 = 27
en2 = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)

p1 = GPIO.PWM(en1, 1000)
p2 = GPIO.PWM(en2, 1000)
p1.start(25)
p2.start(25)

def move_forward():
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)

def move_backward():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)

def stop():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

# Example usage:
try:
    while True:
        command = input("Enter command: ")
        if command == 'f':
            move_forward()
        elif command == 'b':
            move_backward()
        elif command == 's':
            stop()
        elif command == 'l':
            p1.ChangeDutyCycle(25)
            p2.ChangeDutyCycle(25)
        elif command == 'm':
            p1.ChangeDutyCycle(50)
            p2.ChangeDutyCycle(50)
        elif command == 'h':
            p1.ChangeDutyCycle(75)
            p2.ChangeDutyCycle(75)
        elif command == 'e':
            GPIO.cleanup()
            break
        else:
            print("Invalid command")
except KeyboardInterrupt:
    GPIO.cleanup()