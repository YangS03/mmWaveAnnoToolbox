from easydict import EasyDict as edict

const = edict({
    "light_speed": 2.99792458e8,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
})

mmwave = edict({
    "num_lanes": 2,
    "num_tx": 3,
    "num_rx": 4,
    "num_chirps": 64, # change if using FDM          
    "num_frames": 50,
    "num_fast_samples": 256,
    "num_slow_samples": 64, # change if using FDM
    "fs": 640, # change if using FDM
    "adc_sample_bits": 16,
    "freq_slope": 47.990,  # MHz/us
    "adc_sample_rate": 3430,  # ksps
})

mmwave.file_size = mmwave.num_chirps * \
    mmwave.num_frames * mmwave.num_fast_samples * \
        mmwave.num_rx * mmwave.num_tx * mmwave.num_lanes 

mmwave.num_range_bins = 256
mmwave.num_angle_bins = 256
mmwave.num_elevation_bins = 8
mmwave.num_doppler_bins = 64

mmwave.num_antenna = mmwave.num_rx * mmwave.num_tx

mmwave.adc_shape = (mmwave.num_frames, mmwave.num_chirps, mmwave.num_rx * mmwave.num_tx, mmwave.num_fast_samples)
mmwave._chirp_duration = mmwave.num_fast_samples / (mmwave.adc_sample_rate * const.K) # s
mmwave.bandwidth = const.M * mmwave.freq_slope * mmwave._chirp_duration # MHz
mmwave.range_resolution = const.light_speed / (2 * mmwave.bandwidth * const.M) 
mmwave.angle_resolution = 3.1415926 / mmwave.num_angle_bins
mmwave._freq_center = 77 + mmwave.bandwidth * const.M / const.G / 2  # GHz
mmwave._wave_length = const.light_speed / (mmwave._freq_center * const.G) # m
mmwave.doppler_resolution = mmwave._wave_length / (2 * mmwave.num_chirps * mmwave._chirp_duration) # m/s 

radar = {
    'dfeDataOutputMode': '1', 
    'channelCfg': '15 7 0', 
    'adcCfg': '2 1', 
    'adcbufCfg': '-1 0 1 1 1', 
    'profileCfg': '0 77 267 7 57.14 0 0 70 1 256 5209 0 0 30', 
    'chirpCfg': ['0 0 0 0 0 0 0 1', '1 1 0 0 0 0 0 4', '2 2 0 0 0 0 0 2'], 
    'frameCfg': '0 2 64 0 100.0 1 0', 
    # 'frameCfg': '0 2 64 50 100.0 1 0', 
    'lowPower': '0 1', 
    'guiMonitor': '-1 1 1 0 0 0 1', 
    'cfarCfg': ['-1 0 2 8 4 3 0 15 1', '-1 1 0 4 2 3 1 15 1'], 
    'multiObjBeamForming': '-1 1 0.5', 
    'clutterRemoval': '-1 0', 
    'calibDcRangeSig': '-1 0 -5 8 256', 
    'extendedMaxVelocity': '-1 0', 
    'lvdsStreamCfg': '-1 0 1 0', 
    'compRangeBiasAndRxChanPhase': '0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0', 
    'measureRangeBiasAndRxChanPhase': '0 1.5 0.2', 
    'CQRxSatMonitor': '0 3 5 121 0', 
    'CQSigImgMonitor': '0 127 4', 
    'analogMonitor': '0 0', 
    'aoaFovCfg': '-1 -90 90 -90 90', 
    'cfarFovCfg': ['-1 0 0 8.92', '-1 1 -1 1.0'], 
}


if __name__ == '__main__': 
    print('File size:', mmwave.file_size / 1024, 'KB')
    print('Range resolution:', mmwave.range_resolution, 'm')