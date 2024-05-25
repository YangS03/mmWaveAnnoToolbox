import sys
sys.path.append('.')
import cupy as np
# import numpy as np
from copy import deepcopy
from easydict import EasyDict as edict
from numpy.fft import fft, fftshift
from mmRadar.beamformer import BartlettBeamformer, CaponBeamformer

class FMCWRadar(object): 
    def __init__(self, config):
        self.config = edict(config)
        
        assert self.config.num_antenna == 12, "Only support iwrxx43 radars"
        self.beamformer = CaponBeamformer(
            self.config.num_angle_bins, self.config.num_rx * 2)
        
        self.num_frames = self.config.num_frames
        self.num_chirps = self.config.num_chirps
        self.num_antenna = self.config.num_antenna
        self.num_slow_samples = self.config.num_slow_samples  # num_chirps * num_multiframes
        self.num_fast_samples = self.config.num_fast_samples
        self.num_multiframes = self.config.num_slow_samples // self.config.num_chirps
        
        self.num_angle_bins = self.config.num_angle_bins
        self.num_elevation_bins = self.config.num_elevation_bins
        self.num_range_bins = self.config.num_range_bins
        self.num_doppler_bins = self.config.num_doppler_bins
        
        self.wind_func = np.hamming

        
    def read_data(self, filename, complex=False):
        
        assert self.config.adc_sample_bits == 16
        
        f = open(filename, 'rb')
        adc_data = np.fromfile(f, dtype=np.int16)
        filesize = adc_data.shape[0]
        
        assert filesize == self.config.file_size, \
            "Real filesize: " + str(filesize) + " Enpected filesize: " + str(self.config.file_size)
        
        # swap - must deep copy; adc_data: [I1, Q1, I2, Q2, ..., In, Qn]
        # adc_data_1 = deepcopy(adc_data[1: filesize: 4])
        # adc_data_2 = deepcopy(adc_data[2: filesize: 4])
        # adc_data[2: filesize: 4] = adc_data_1
        # adc_data[1: filesize: 4] = adc_data_2
        
        # print("filesize: ", filesize)
        # print("num_frames: ", self.config.num_frames)
        # print("num_chirps: ", self.config.num_chirps)
        # print("num_antenna: ", self.config.num_antenna)
        # print("num_samples: ", self.config.num_fast_samples)
        
        adc_data_I = adc_data[0: filesize: 2]
        adc_data_Q = adc_data[1: filesize: 2]
        
        # ATTENTION: data capture using API, I and Q are swapped
        adc_data_I_ = deepcopy(adc_data_Q)
        adc_data_Q_ = deepcopy(adc_data_I)
        
        # adc_data_I_ = deepcopy(adc_data_I)
        # adc_data_Q_ = deepcopy(adc_data_Q)
        
        # [num_frames, num_chirps, num_antenna, num_samples]
        adc_data_I_ = np.reshape(adc_data_I_, self.config.adc_shape)
        adc_data_Q_ = np.reshape(adc_data_Q_, self.config.adc_shape)
        
        if not complex:      
            return adc_data_I_, adc_data_Q_
        else: 
            return adc_data_I_ + 1j * adc_data_Q_
    
    def read_stream(self, data_stream):
        data_frame = np.reshape(data_stream, (self.num_chirps, self.num_antenna, self.num_fast_samples))
        return data_frame
        
    def range_fft(self, IQ_complex, target_frame_idx=None, num_multiframe=1):
        if target_frame_idx is not None:
            # Using cupy to perform fft
            IQ_complex = IQ_complex[target_frame_idx: target_frame_idx+num_multiframe, :, :, :] # [num_multi_frames, num_chirps, num_antenna, num_fast_samples]
            IQ_samples = np.reshape(IQ_complex, (self.num_slow_samples, self.num_antenna, self.num_fast_samples))   # [num_slow_samples, num_antenna, num_fast_samples]
            IQ_samples = IQ_samples * self.wind_func(self.num_fast_samples)
            slow_time_samples = np.fft.fft(IQ_samples, n=self.num_range_bins, axis=-1)
            # [num_range_bins, num_antenna, num_slow_samples]
            slow_time_samples = np.transpose(slow_time_samples[:, :, :], (2, 1, 0))
        else: 
            # Using cupy to perform fft
            IQ_samples = IQ_complex * self.wind_func(self.num_fast_samples)  
            slow_time_samples = np.fft.fft(IQ_samples, n=self.num_range_bins, axis=-1)
            slow_time_samples = np.transpose(slow_time_samples[:, :, :], (2, 1, 0))
        # assert slow_time_samples.shape == (self.num_range_bins, self.num_antenna, self.num_slow_samples)
        return slow_time_samples
    
    def doppler_fft(self, slow_time_samples):
        # Using cupy to perform fft
        # [num_range_bins, num_antenna, num_dopper_bins]
        slow_time_samples = slow_time_samples * self.wind_func(self.num_doppler_bins)
        doppler_samples = np.fft.fft(slow_time_samples, n=self.num_doppler_bins, axis=-1)
        doppler_samples = np.fft.fftshift(doppler_samples, axes=-1)
        return doppler_samples
    
    def phase_error_compensation(self, slow_time_samples):
        # Compensate phase error for virtual antennas
        # TODO: Velocity compensation, currently only phase compensation
        # TODO: Check the correctness of phase compensation
        range_doppler_spectrum = self.doppler_fft(slow_time_samples)  # [num_range_bins, num_antenna, num_doppler_bins]
        max_doppler_bins = np.argmax(np.abs(range_doppler_spectrum), axis=-1)   # [num_range_bins, num_antenna]
        
        phi_error = 2 * np.pi * (max_doppler_bins / self.num_slow_samples - 0.5)    # [num_range_bins, num_antenna, num_slow_samples]
        
        for i in range(0, self.config.num_tx): 
            slow_time_samples[:, i * self.config.num_rx: (i + 1) * self.config.num_rx, :] *= \
                np.exp(-1j * i * phi_error[:, i * self.config.num_rx: (i + 1) * self.config.num_rx][:, :, np.newaxis])
                
        return slow_time_samples
    
    def parse_data(self, IQ_complex): 
        # [num_range_bins, num_antenna, num_chirps]
        rx = self.config.num_rx
        radar_data_8rx = IQ_complex[:, :rx * 2, :]
        radar_data_4rx = IQ_complex[:, rx * 2:, :]
        return radar_data_8rx, radar_data_4rx
        
    def get_window_sample(self, slow_time_samples, window_size, window_step=1, sample_mothod="sliding"):
        # slow_time_samples: [num_range_bins, num_antenna, num_slow_samples]
        # Get window samples
        if sample_mothod == "sliding":
            num_window_samples = (self.config.num_slow_samples  - window_size) // window_step + 1
            window_slow_time_samples = np.zeros((num_window_samples, self.config.num_fast_samples, self.config.num_antenna, window_size), dtype=np.complex64)
            for idx_wind in range(num_window_samples):
                sub_slow_time_samples = slow_time_samples[:, :, idx_wind * window_step: idx_wind * window_step + window_size]
                window_slow_time_samples[idx_wind, :, :, :] = sub_slow_time_samples    
        return window_slow_time_samples
            
    def angle_fft(self, slow_time_samples):
        # Using cupy to perform fft
        # slow_time_samples: [num_range_bins, num_antenna, num_dopper_bins]
        num_antenna = slow_time_samples.shape[1]
        slow_time_samples = slow_time_samples * self.wind_func(num_antenna)[np.newaxis, :, np.newaxis]
        angle_samples = np.fft.fft(slow_time_samples, n=self.num_angle_bins, axis=1)
        angle_samples = np.fft.fftshift(angle_samples, axes=1)
        return angle_samples
    
    def elevation_fft(self, slow_time_samples):
        # Using cupy to perform fft
        # slow_time_samples: [num_range_bins, num_angle_bins, num_elevate, num_dopper_bins]
        angle_samples = np.fft.fft(slow_time_samples, n=self.num_elevation_bins, axis=2)
        angle_samples = np.fft.fftshift(angle_samples, axes=2)
        return angle_samples
    
    def remove_direct_component(self, range_ffts):
        num_slow_time_samples = range_ffts.shape[-1]
        shape_len = len(range_ffts.shape)

        if shape_len == 3:
            mean_vals = np.repeat(np.mean(range_ffts, axis=2)[:, :, np.newaxis], num_slow_time_samples, axis=2)
            return range_ffts - mean_vals
        elif shape_len == 4:
            mean_vals = np.repeat(np.mean(range_ffts, axis=3)[:, :, :, np.newaxis], num_slow_time_samples, axis=3)
            return range_ffts - mean_vals
        else:
            assert 1 == 0, "Wrong dimension to remove DC!!!"

    def perform_beamforming(self, slow_time_samples, update_gain_mat=False):
        """
            input: 
                slow_time_samples: [num_range_bins, num_antenna, num_slow_samples]
                update_gain_mat: whether to update the gain matrix
            description:
                perform beamforming in range-angle domain
        """
        self.num_range_bins = slow_time_samples.shape[0]
        
        if update_gain_mat:
            self.update_RCB_gains(slow_time_samples)
        
        power_spectrums, beamforming_weights = self.beamformer.steering(slow_time_samples)
        
        return power_spectrums, beamforming_weights
    
    def update_RCB_gains(self, multi_rx_slow_time_samples, ratio=95):

        # Update Robust Capon Beamforming Gain Guess
        rs = np.abs(multi_rx_slow_time_samples)
        strengths = np.percentile(rs, ratio, axis=-1)   # [num_range_bins, num_antennas]
        
        self.beamformer.update_gain_mat(strengths)

    def extract_with_beamforming(self, slow_time_samples, beamforming_weights, range_bin_idx=None, angle_bin_idx=None):
        
        # slow_time_samples: [num_range_bins, num_antenna, num_slow_samples]
        # beamforming_weights: [num_range_bins, num_antenna, num_angle_bins]
        beamforming_weights_H = np.conj(beamforming_weights).transpose(0, 2, 1) # [num_range_bins, num_angle_bins, num_antenna]
        slow_time_signal = np.einsum('ijk,ikl->ijl', beamforming_weights_H, slow_time_samples)  # [num_range_bins, num_angle_bins, num_slow_samples]
        
        return slow_time_signal 

    def get_spectrum_data(self, slow_time_samples, method='beamform'): 
        slow_time_samples = self.remove_direct_component(slow_time_samples)
        
        if method == 'beamform':
            # Get angle data
            range_angle_spectrum, _ = self.perform_beamforming(slow_time_samples, update_gain_mat=False)
            range_angle_spectrum = np.abs(range_angle_spectrum)
            return range_angle_spectrum
        elif method == 'fft':
            # Get doppler data
            doppler_samples = self.doppler_fft(slow_time_samples)
            range_angle_spectrum = self.angle_fft(doppler_samples)
            range_angle_spectrum = np.abs(range_angle_spectrum)
            range_angle_spectrum = range_angle_spectrum.sum(axis=-1)
        return range_angle_spectrum

    def get_RAED_data(self, radar_data_8rx, radar_data_4rx): 
        # Get range data
        radar_data_8rx = self.range_fft(radar_data_8rx)
        radar_data_4rx = self.range_fft(radar_data_4rx)
        # Remove direct component
        radar_data_8rx = self.remove_direct_component(radar_data_8rx)
        radar_data_4rx = self.remove_direct_component(radar_data_4rx)
        # Get doppler data
        radar_data_8rx = self.doppler_fft(radar_data_8rx)
        radar_data_4rx = self.doppler_fft(radar_data_4rx)
        # Get angle data
        radar_data_8rx = self.angle_fft(radar_data_8rx)
        radar_data_4rx = self.angle_fft(radar_data_4rx)
        radar_data = np.stack([radar_data_8rx, radar_data_4rx], axis=2)
        # Get elevation data
        radar_data = self.elevation_fft(radar_data)
        radar_data = np.abs(radar_data)
        return radar_data