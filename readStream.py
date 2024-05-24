import time
import cupy as cp
import numpy as np
import mmwave as mm
import open3d as o3d
from mmwave.dataloader import DCA1000
from radar.fmcw_radar import FMCWRadar
from pcGenerator.pc_generator import get_pointcloud
import radar_configs.mmwave_radar.iwr1843 as cfg
from matplotlib import pyplot as plt

if __name__ == "__main__": 
    dca = DCA1000(static_ip='192.168.33.31', adc_ip='192.168.33.181', data_port=4088, config_port=4086)
    radar = FMCWRadar(cfg)
    plt.ion()
    
    while True: 
        adc_data = dca.read()
        data_frame = dca.organize(adc_data, cfg.mmwave.num_slow_samples, cfg.mmwave.num_antenna, cfg.mmwave.num_fast_samples)
        tic = time.time()
        
        data_frame = cp.array(data_frame)
        data_frame = radar.range_fft(data_frame)
        radar_data_8rx, radar_data_4rx = radar.parse_data(data_frame)
        ra_map = radar.get_spectrum_data(radar_data_8rx, method='beamform')

        plt.clf()
        plt.imshow(np.abs(ra_map.get()))
        plt.pause(0.01)
        print('Time taken: ', time.time() - tic)
        
        