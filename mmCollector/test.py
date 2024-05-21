import serial

ser = serial.Serial('COM23', 115200, timeout=1)
while True:
    # ser.write(b'123\n')
    a = ser.readline()