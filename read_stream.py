import time
import cupy as cp
import numpy as np
import mmwave as mm
from mmwave.dataloader import DCA1000
import radar_configs.mmwave_radar.iwr1843 as cfg
import matplotlib.pyplot as plt
from radar.fmcw_radar import FMCWRadar

if __name__ == "__main__": 
    dca = DCA1000(static_ip='192.168.33.31', adc_ip='192.168.33.181', data_port=4088, config_port=4086)
    radar = FMCWRadar(cfg)

    plt.ion()

    while True: 
        adc_data = dca.read()
        radar_complex = dca.organize(adc_data, cfg.mmwave.num_slow_samples, cfg.mmwave.num_antenna, cfg.mmwave.num_fast_samples)
        
        tic = time.time()
        # perform range fft
        radar_complex = cp.array(radar_complex)
        radar_complex = radar.range_fft(radar_complex)
        # parse channels
        radar_data_8rx, radar_data_4rx = radar.parse_data(radar_complex)
        # perfrom beamforming
        radar_spec = radar.get_spectrum_data(radar_data_8rx, type='beamforming')
        radar_spec = np.flip(radar_spec, axis=0)
        print('Time taken: ', time.time() - tic)
        
        plt.clf()
        plt.imshow(radar_spec.get(), origin='lower')
        plt.show()
        plt.pause(0.01)
        