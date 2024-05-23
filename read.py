import numpy as np
import cupy as cp
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
        # adc_data = mm.dsp.clutter_removal(adc_data)
        data_mat = dca.organize(adc_data, cfg.mmwave.num_slow_samples, cfg.mmwave.num_antenna, cfg.mmwave.num_fast_samples)
        
        data_mat = cp.array(data_mat)
        slow_time_samples = radar.range_fft(data_mat)
        slow_time_samples = radar.remove_direct_component(slow_time_samples)
        slow_time_samples = radar.doppler_fft(slow_time_samples)
        range_angle_spectrum = radar.angle_fft(slow_time_samples)
        
        plt.imshow(np.abs(range_angle_spectrum).sum(2).get(), origin='lower')
        plt.show()
        plt.pause(0.01)
        