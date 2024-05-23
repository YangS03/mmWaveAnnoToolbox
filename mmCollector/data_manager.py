
import os
os.chdir('.')
import sys
sys.path.append('.')
import numpy as np
from radar_configs.mmwave_radar import iwr1843


def parse_date_info(date_info):
    month_to_num = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }
    month_digit = month_to_num[date_info[0]]

    day = date_info[1]
    time = date_info[2].replace(':', '-')
    year = date_info[3].rstrip('\n')

    date_seq = f"{year}-{month_digit}-{day}-{time}"
    
    return date_seq


def read_data(id=None):
    assert id in ['vert', 'hori'], 'Invalid id'
    adc_file_path = r'D:\Nutstore\Workspace\mmWaveAnnoToolbox\dca1000\saved_files\{}\adc_data_Raw_0.bin'.format(id)
    log_file_path = r'D:\Nutstore\Workspace\mmWaveAnnoToolbox\dca1000\saved_files\{}\adc_data_Raw_LogFile.csv'.format(id)
    file_stat = os.stat(adc_file_path)
    with open(adc_file_path, 'rb') as f:
        adc_data = np.fromfile(f, dtype=np.int16)
        
    with open(log_file_path, 'r') as f:
        log_data = f.readlines()
        capture_start_info = log_data[-3].split(' ')[-4:]
        capture_end_info = log_data[-2].split(' ')[-4:]
        duration = log_data[-1].split(' ')[-1].rstrip('\n')
        
    
    start_time_info = parse_date_info(capture_start_info)
    end_time_info = parse_date_info(capture_end_info) 
    
    return {
        'file_size': file_stat.st_size / 1024,
        'start_time': start_time_info,
        'end_time': end_time_info,
        'duration': duration,
    }


if __name__ == '__main__': 
    info = read_data('vert')
    print(info['file_size'])