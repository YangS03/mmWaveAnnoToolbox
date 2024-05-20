from easydict import EasyDict as edict

const = edict({
    "light_speed": 2.99792458e8,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
})

mmwave = edict({
    "num_lanes": 2,
    "num_tx": 2,
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
        mmwave.num_rx * mmwave.num_tx * mmwave.num_lanes * 2

mmwave.num_range_bins = 256
mmwave.num_angle_bins = 256
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

radar = edict({
    # 1: frame based chirps
    # 2: continuous chirping (not supported)
    # 3: advanced frame config
    "dfeDataOutputMode": 1,
    # Channel configuration
    # "rxChanEn": Bitmask for enabling RX channels (1: Enable, 0: Disable)
    # "txChanEn": Bitmask for enabling TX channels (1: Enable, 0: Disable)
    # "cascading": Cascading mode (0: No cascading, 1: Cascading enabled)
    "channelCfg": {
        '15 7 0'
    },
    # ADC configuration
    # "numADCBits": ADC resolution (0: 12-bit, 1: 14-bit, 2: 16-bit)
    # "adcOutFmt": ADC output format (0: Real, 1: Complex1, 2: Complex2)
    "adcCfg": {
        '2 1'
    },
    # ADC buffer configuration
    "adcbufCfg": {
        '-1 0 1 1 1'
    },
    # Profile configuration
    "profileCfg": {
        '0 77 267 7 57.14 0 0 70 1 256 5209 0 0 30'
    },
    # Chirp configuration
    "chirpCfg": [{
        '1 1 0 0 0 0 0 4'
    }, {
        '2 2 0 0 0 0 0 2'
    }], 
    # Frame configuration
    "frameCfg": {
        '0 2 16 0 100 1 0'
    },
    # Low power mode configuration
    "lowPower": {
        '0 0'
    },
    # GUI monitor configuration (do not change)
    "guiMonitor": {
        '-1 1 1 0 0 0 1'
    },
    # CFAR configuration (do not change)
    "cfarCfg": [{
        '-1 0 2 8 4 3 0 15 1'
    }, {
        '-1 1 0 4 2 3 1 15 1'
    }], 
    # Multi-object beam forming configuration (do not change)
    "multiObjBeamForming": {
        '-1 1 0.5'
    },
    # Clutter removal configuration (do not change)
    "clutterRemoval": {
        '-1 0'
    },
    # Calibration DC range signature configuration (do not change)
    "calibDcRangeSig": {
        '-1 0 0 0'
    }, 
    # Extended maximum velocity configuration (do not change)
    "extendedMaxVelocity": {
        "subFrameIdx": -1,
        "enable": 0
    },
    # BPM configuration (do not change)
    "bpmCfg": {
        "subFrameIdx": -1,
        "enable": 0, 
        "chirp0Idx": 0, 
        "chirp1Idx": 1
    },
    # LVDS stream configuration
    # "subFrameIdx": Sub-frame index (-1: Default)
    # "enableHeader": Enable header flag (0: Disable, 1: Enable)
    # "dataFmt": Data format (0: HW disable, 1: ADC, 2: CP_ADC_CQ)
    # "enableSW": Enable software flag (0: Disable, 1: Enable)
    "lvdsStreamCfg": {
        "subFrameIdx": -1,
        "enableHeader": 0,
        "dataFmt": 1,
        "enableSW": 0
    },
    # Range bias and RX channel phase compensation configuration
    # "rangeBias": Range bias value
    # "I/Q Bias compen for 2Tx*4Rx": I/Q bias compensation for 2Tx*4Rx channels
    "compRangeBiasAndRxChanPhase": "0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0", 
    # Measure range bias and RX channel phase configuration
    "measureRangeBiasAndRxChanPhase": {
        "ensable": 0,
        "targetDistance": 1.5,
        "searchWin": 0.2
    },
    # CQ RX saturation monitor configuration
    "CQRxSatMonitor": {
        "profile": 0,
        "satMonSel": 3,
        "priSliceDuration": 5, 
        "numSlices": 121,
        "rxChanMask": 0, 
    },
    # CQ signal image monitor configuration
    "CQSigImgMonitor": {
        "profile": 0,
        "numSlices": 127,
        "numSamplePerSlice": 4
    },
    # Analog monitor configuration
    "analogMonitor": {
        "enable": 0,
        "sigImgBand": 0
    },
    # AOA field of view configuration (do not change)
    "aoaFovCfg": {
        "subFrameIdx": -1,
        "minAzimuthDeg": -90,
        "maxAzimuthDeg": 90,
        "minElevationDeg": -90,
        "maxElevationDeg": 90
    },
    # CFAR field of view configuration
    "cfarFovCfg": [{
        "subFrameIdx": -1,
        "procDirection": 0, #  point filtering in range direction
        "min": 0,
        "max": 8.92
    }, {
        "subFrameIdx": -1,
        "procDirection": 1, # point filtering in Doppler
        "min": -1,
        "max": 1.00
    }], 
    "calibData": "0 0 0"
})


if __name__ == '__main__': 
    print('File size:', mmwave.file_size / 1024, 'KB')
    print('Range resolution:', mmwave.range_resolution, 'm')