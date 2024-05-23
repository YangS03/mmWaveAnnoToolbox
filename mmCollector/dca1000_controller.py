import sys
sys.path.append('.')
import os
os.chdir('.\dca1000')


def readline_from_process(process):
    lines = ''
    while True:
        line = process.readline()
        if line == '\n': 
            continue
        if not line:
            break
        lines += line
    return lines


def configure_fpga(id=None): 
    assert id in ['vert', 'hori'], 'Invalid id'
    command_configure_fpga = '.\DCA1000EVM_CLI_Control.exe fpga .\cf_{}.json'.format(id)
    with os.popen(command_configure_fpga) as p: 
        return readline_from_process(p)
        

def reset_fpga(id=None):
    assert id in ['vert', 'hori'], 'Invalid id'
    command_reset_fpga = '.\DCA1000EVM_CLI_Control.exe reset_fpga .\cf_{}.json'.format(id)
    with os.popen(command_reset_fpga) as p: 
        return readline_from_process(p)


def start_record(id=None): 
    assert id in ['vert', 'hori'], 'Invalid id'
    command_start_record = '.\DCA1000EVM_CLI_Record.exe start_record .\cf_{}.json'.format(id)
    with os.popen(command_start_record) as p: 
        return readline_from_process(p)


if __name__ == '__main__': 
    out = reset_fpga('vert')
    out = configure_fpga('vert')
    # out = start_record('vert')
    print(out)