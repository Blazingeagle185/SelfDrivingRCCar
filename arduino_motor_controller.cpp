// arduino_motor_controller.ino
// Arduino sketch for RC car ESC and steering control via serial

#include <Servo.h>

Servo esc;
Servo steering;

const int escPin = 9;
const int steerPin = 10;
unsigned long lastCommandTime = 0;
const unsigned long timeout = 2000;  // stop after 2 seconds of inactivity

void setup() {
  Serial.begin(9600);
  esc.attach(escPin);
  steering.attach(steerPin);
  stopMotors();
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    handleCommand(command);
    lastCommandTime = millis();
  }

  if (millis() - lastCommandTime > timeout) {
    stopMotors();
  }
}

void handleCommand(char cmd) {
  switch (cmd) {
    case 'F': // forward
      esc.write(120);
      break;
    case 'L': // left
      steering.write(60);
      break;
    case 'R': // right
      steering.write(120);
      break;
    case 'B': // brake/reverse
      esc.write(60);
      break;
    case 'S': // stop
      stopMotors();
      break;
  }
}

void stopMotors() {
  esc.write(90);       // neutral signal
  steering.write(90);  // center steering
} 
