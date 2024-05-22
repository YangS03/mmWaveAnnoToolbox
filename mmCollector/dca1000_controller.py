import sys
sys.path.append('.')
import os
os.chdir('.\dca1000')



def readline_from_process(process):
    while True:
        line = process.readline()
        if line == '\n': 
            continue
        if not line:
            break
        print(line, end='')


def configure_fpga(id=None): 
    assert id in ['vert', 'hori'], 'Invalid id'
    command_configure_fpga = '.\DCA1000EVM_CLI_Control.exe fpga .\cf_{}.json'.format(id)
    with os.popen(command_configure_fpga) as p: 
        readline_from_process(p)
        

def reset_fpga(id=None):
    assert id in ['vert', 'hori'], 'Invalid id'
    command_reset_fpga = '.\DCA1000EVM_CLI_Control.exe reset_fpga .\cf_{}.json'.format(id)
    with os.popen(command_reset_fpga) as p: 
        readline_from_process(p)


def start_record(id=None): 
    command_start_record = '.\DCA1000EVM_CLI_Record.exe start_record .\cf_{}.json'.format(id)
    with os.popen(command_start_record) as p: 
        readline_from_process(p)


if __name__ == '__main__': 
    reset_fpga('vert')
    configure_fpga('vert')
    start_record('vert')