import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from mmRadar import FMCWRadar
from mmRadar.utils.convert import dB
import radarConfigs.mmwave_radar.iwr1843 as cfg

if __name__ == '__main__': 

    plt.ion()

    # create radar object
    radar = FMCWRadar(cfg.mmwave)
    bin_filename = r"E:\CollectedData\wxg_test\2024-05-24-17-18-12\adc_data_hori.bin"
    bin_data = radar.read_data(bin_filename, complex=True)

    for idx_frame in range(cfg.mmwave.num_frames):
        data_frame = bin_data[idx_frame]    # [num_chirps, num_attennas, num_adc_samples]
        # perform range fft
        data_frame = radar.remove_direct_component(data_frame, axis=0)
        data_frame = radar.range_fft(data_frame)
        
        radar_data_8rx, radar_data_4rx = radar.parse_data(data_frame)
        ra_map = radar.get_spectrum_data(radar_data_8rx, method='fft')
        ra_map = dB(ra_map)
        
        plt.clf()
        plt.imshow(np.abs(ra_map.get()))
        plt.pause(0.01)
        
    