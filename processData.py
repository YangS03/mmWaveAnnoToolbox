import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import radar_configs.mmwave_radar.iwr1843 as cfg
from utils.convert import dB
from radar.fmcw_radar import FMCWRadar
from radar.beamformer import BartlettBeamformer, CaponBeamformer

if __name__ == '__main__': 

    plt.ion()

    # create radar object
    radar = FMCWRadar(cfg)
    bin_filename = r"E:\CollectedData\wxg_test\2024-05-24-17-18-12\adc_data_hori.bin"
    bin_data = radar.read_data(bin_filename, complex=True)

    for idx_frame in range(cfg.mmwave.num_frames):
        data_frame = bin_data[idx_frame]    # [num_chirps, num_attennas, num_adc_samples]
        # perform range fft
        data_frame = radar.range_fft(data_frame)
        data_frame = radar.remove_direct_component(data_frame)
        
        radar_data_8rx, radar_data_4rx = radar.parse_data(data_frame)
        # radar_4d_heatmap = radar.get_RAED_data(radar_data_8rx, radar_data_4rx)  # [range, angle, elevation, doppler]
        # rd_map = radar.doppler_fft(radar_data_8rx)
        # ra_map = dB(ra_map)
        ra_map = radar.get_spectrum_data(radar_data_8rx, method='beamform')
        plt.clf()
        plt.imshow(np.abs(ra_map.get()))
        plt.pause(0.1)
        
    