import cmd
import time
import importlib
import serial
import serial.tools.list_ports as list_ports

class mmWaveRecoder:
    def __init__(self, port, baudrate=115200, device='iwr1843'):
        self.port = port
        self.baudrate = baudrate
        self.device = device
        
        self.serial = None
        self.serial_data = ['sensorStop', 'flushCfg']
        
        self._load_config()
        self._connect_serial()
        self.send_config()
        
        self.first_frame = True
        
    def _load_config(self):
        cfg = importlib.import_module('configs.mmwave_radar.' + self.device)
        for cfg_key, cfg_value in cfg.radar.items():
            if isinstance(cfg_value, str): 
                cfg_item = cfg_key + ' ' + cfg_value
                self.serial_data.append(cfg_item)    
            elif isinstance(cfg_value, list):
                for sub_item in cfg_value:
                    cfg_item = cfg_key + ' ' + sub_item
                    self.serial_data.append(cfg_item)    
            else: 
                raise ValueError('Invalid configuration')
            
    def _connect_serial(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
    
    def _write_serial(self, data):
        if self.serial is not None: 
            self.serial.write((data + '\n').encode())
        time.sleep(0.01)

    def show_config(self):
        for data in self.serial_data:
            print(data)
            time.sleep(0.01)
    
    def send_config(self): 
        for data in self.serial_data:
            self._write_serial(data)
    
    def send_start(self):
        if self.first_frame: 
            self._write_serial('sensorStart')        
            
        
    def send_stop(self):
        self._write_serial('sensorStop')    
        
        
class CMDClient(cmd.Cmd):
    def __init__(self):
        super(CMDClient, self).__init__()
        # search for available ports
        ports = list_ports.comports()
        print('Available ports:')
        if len(ports) == 0:
            print('No available ports')
        for port, desc, hwid in sorted(ports):
            print("{}: {}".format(port, desc))
        # initialize the recorders
        port = 'COM' + input('Please select a port:')        
        self.recorder = mmWaveRecoder(port=port)
    
    def do_start_record(self, arg):
        self.recorder.send_start()
        
    def do_stop_record(self, arg):
        self.recorder.send_stop()
        
    def do_show_config(self, arg):
        self.recorder.show_config()
        
    def do_quit(self, arg):
        'Quit the application'
        print('Exiting...')
        return True
    
if __name__ == '__main__': 
    cmd = CMDClient()
    cmd.cmdloop()