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
        "rxChanEn": 0b1111,  # Enable all 4 RX channels
        "txChanEn": 0b01 if mmwave.num_tx==1 else 0b11,   # Enable TX channel 1
        "cascading": 0,      # No cascading
    },
    # ADC configuration
    # "numADCBits": ADC resolution (0: 12-bit, 1: 14-bit, 2: 16-bit)
    # "adcOutFmt": ADC output format (0: Real, 1: Complex1, 2: Complex2)
    "adcCfg": {
        "numADCBits": 2,     # 2: 16-bit ADC resolution
        "adcOutFmt": 1,      # 1: Complex1 output format
    },
    # ADC buffer configuration
    "adcbufCfg": {
        "subFrameIdx": -1,   # Sub-frame index: -1 for default
        "adcOutFmt": 0,      # 0: Complex output format
        "sampleSwap": 1,      # 1: I in MSB, Q in LSB
        "ChanInterleave": 1,  # 1: Non-interleaved data
        "ChirpThreshold": 1,  # 1: LVDS chirp threshold
    },
    # Profile configuration
    "profileCfg": {
        "profID": 0,
        "startFreq": 77,       # GHz
        "idleTime": 420,       # us
        "adcStartTime": 5,     # us
        "rampEndTime": 80.0,     # us
        "txOutPower": 0,       # dBm
        "txPhaseShift": 0,     # degrees
        "freqSlopeConst": mmwave.freq_slope,  # MHz/us
        "txStartTime": 0,      # us
        "numAdcSample": mmwave.num_fast_samples,
        "digOutSampleRate": mmwave.adc_sample_rate,  # ksps
        "hpfCornerFreq1": 0,   # 0: 175KHz, 1: 235KHz, 2: 350KHz, 3: 700KHz
        "hpfCornerFreq2": 0,   # 0: 350KHz, 1: 700KHz, 2: 1400KHz, 3: 2800KHz
        "rxGain": 30,          # dB
    },
    # Chirp configuration
    "chirpCfg": [{
        "startIdx": 0,
        "endIdx": 0,
        "profID": 0,
        "startFreqVar": 0,
        "freqSlopeVar": 0,
        "idleTimeVar": 0,
        "AdcStartTimeVar": 0,
        "txEnableMask": 0b01,
    }, {
        "startIdx": 1,
        "endIdx": 1,
        "profID": 0,
        "startFreqVar": 0,
        "freqSlopeVar": 0,
        "idleTimeVar": 0,
        "AdcStartTimeVar": 0,
        "txEnableMask": 0b10,
    }][0: mmwave.num_tx], 
    # Frame configuration
    "frameCfg": {
        "startIdx": 0,
        "endIdx": 1 if mmwave.num_tx==2 else 0,
        "loopNum": mmwave.num_chirps,
        "frameNum": mmwave.num_frames,
        "framePerio": mmwave.num_chirps/(mmwave.fs) * 1000,
        "trigSel": 1,
        "frameTrigDelay": 0,
    },
    # Low power mode configuration
    "lowPower": {
        "Ignored": 0,   # Ignored value (0)
        "AdcMode": 1,   # ADC mode (0: Regular, 1: LP Mode)
    },
    # GUI monitor configuration (do not change)
    "guiMonitor": {
        "subFrameIdx": -1,
        "detectedObj": 0,
        "logMagRange": 0,
        "noiseProf": 0,
        "rangeAziHeatmap": 0,
        "rangeDFSHeatmap": 0,
        "stasInfo": 0,
    },
    # CFAR configuration (do not change)
    "cfarCfg": [{
        "subFrameIdx": -1,
        "procDirection": 0,
        "mode": 2,
        "noiseWin": 8,
        "guardLen": 4,
        "divShift": 3,
        "cyclicMode": 0,
        "thresholdScale": 15,
        "peakGrouping": 0
    }, {
        "subFrameIdx": -1,
        "procDirection": 1,
        "mode": 0,
        "noiseWin": 4,
        "guardLen": 2,
        "divShift": 3,
        "cyclicMode": 1,
        "thresholdScale": 15,
        "peakGrouping": 1
    }], 
    # Multi-object beam forming configuration (do not change)
    "multiObjBeamForming": {
        "subFrameIdx": -1,
        "enable": 0,
        "threshold": 0.5
    },
    # Clutter removal configuration (do not change)
    "clutterRemoval": {
        "subFrameIdx": -1,
        "enable": 1
    },
    # Calibration DC range signature configuration (do not change)
    "calibDcRangeSig": {
        "subFrameIdx": -1, 
        "enable": 0, 
        "negativeBinIdx": -5, 
        "positiveBinIdx": 8, 
        "numAvg": 256
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