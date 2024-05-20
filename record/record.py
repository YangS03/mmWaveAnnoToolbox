import serial, time
from easydict import EasyDict as edict
from mmwave_cfg import radar as cfg
# Configure IWR1642 by serial port
enable_CLI_1 = True
enable_CLI_2 = False
serial_port_CLI_1 = 'COM18'
serial_port_CLI_2 = 'COM14'
file_cfg = 'cfg/profile_old.cfg'
if enable_CLI_1:
    CLIport_1 = serial.Serial(serial_port_CLI_1, 115200, timeout=5)
if enable_CLI_2: 
    CLIport_2 = serial.Serial(serial_port_CLI_2, 115200, timeout=1)

config = []
config.append("sensorStop")
config.append("flushCfg")
for cfg_item in cfg.items():
    if type(cfg_item[1]) == edict:
        config.append(cfg_item[0] + " " + " ".join(map(str, cfg_item[1].values())))
    elif type(cfg_item[1]) == list:
        for sub_item in cfg_item[1]:
            config.append(cfg_item[0] + " " + " ".join(map(str, sub_item.values())))
    elif type(cfg_item[1]) == str: 
        config.append(cfg_item[0] + " " + cfg_item[1])
    else:
        config.append(cfg_item[0] + " " + str(cfg_item[1]))

for i in config:
    if enable_CLI_1:
        CLIport_1.write((i+'\n').encode())
    if enable_CLI_2:
        CLIport_2.write((i+'\n').encode())
    print('>>> ' + i)
    time.sleep(0.01)

# Wait key to toggle frame
sending = False
initial_frame_sent = False
while True:
    if sending:
        print('\nFrame ' + 'sending' + ', press Enter to ' + 'stop')
        key_input = input('<<')
        # involke stop
        if enable_CLI_1: 
            CLIport_1.write(('sensorStop\n').encode())
        if enable_CLI_2:
            CLIport_2.write(('sensorStop\n').encode())
        print('>>> sensorStop')
        time.sleep(0.01)
        sending = False
    else:
        print('\nFrame ' + 'stopped' + ', press Enter to ' + 'send')
        key_input = input('<<')
        #  involke send
        if initial_frame_sent:
            start_cmd = 'sensorStart 0'
        else:
            start_cmd = 'sensorStart'
        if enable_CLI_1: 
            CLIport_1.write((start_cmd + '\n').encode())
        if enable_CLI_2:
            CLIport_2.write((start_cmd + '\n').encode())
        print('>>> ' + start_cmd)
        time.sleep(0.01)
        sending = True
        initial_frame_sent = True
        