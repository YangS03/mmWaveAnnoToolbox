import sys
sys.path.append('.')
import cupy as np
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
        
        # self.wind_func = np.hamming
        self.wind_func = np.ones

    def read_data(self, filename, complex=False):
        assert self.config.adc_sample_bits == 16, "Only support 16-bit ADC data"
            
        f = open(filename, 'rb')
        adc_data = np.fromfile(f, dtype=np.int16)
        file_size = adc_data.shape[0] // 2
        
        # assert file_size == self.config.file_size, "Real file_size: " + str(file_size) + " Expected file_size: " + str(self.config.file_size)
        error_size = (self.config.file_size - file_size) * 2
        assert abs(error_size) < 1024, "Real file_size: " + str(file_size) + " Expected file_size: " + str(self.config.file_size)
        adc_data = np.pad(adc_data, (0, error_size))
        
        # # swap - must deep copy; adc_data: [I1, I2, Q1, Q2, ...] -> [I1, Q1, I2, Q2, ..., In, Qn]
        # # no need if using DCA1000 CLI 
        # adc_data_1 = deepcopy(adc_data[1:: 4])
        # adc_data_2 = deepcopy(adc_data[2:: 4])
        # adc_data[2:: 4] = adc_data_1
        # adc_data[1:: 4] = adc_data_2
        
        # print("file_size: ", file_size)
        # print("num_frames: ", self.config.num_frames)
        # print("num_chirps: ", self.config.num_chirps)
        # print("num_antenna: ", self.config.num_antenna)
        # print("num_samples: ", self.config.num_fast_samples)
        
        adc_data_I = adc_data[0:: 2]
        adc_data_Q = adc_data[1:: 2]
        
        # ATTENTION: data capture using API, I and Q are swapped (I in MSB, Q in LSB)
        adc_data_I_ = deepcopy(adc_data_Q)
        adc_data_Q_ = deepcopy(adc_data_I)
        
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
            slow_time_samples = np.transpose(slow_time_samples, (2, 1, 0))
        else: 
            # Using cupy to perform fft
            IQ_samples = IQ_complex
            IQ_samples = IQ_samples * self.wind_func(self.num_fast_samples)  
            slow_time_samples = np.fft.fft(IQ_samples, n=self.num_range_bins, axis=-1)
            slow_time_samples = np.transpose(slow_time_samples, (2, 1, 0))
        # assert slow_time_samples.shape == (self.num_range_bins, self.num_antenna, self.num_slow_samples)
        return slow_time_samples
    
    def doppler_fft(self, slow_time_samples, axis=-1, shift=True):
        # Using cupy to perform fft
        # [num_range_bins, num_antenna, num_dopper_bins]
        doppler_samples = np.fft.fft(slow_time_samples, n=self.num_doppler_bins, axis=axis)
        if shift: 
            doppler_samples = np.fft.fftshift(doppler_samples, axes=axis)
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
        # radar_data_8rx = IQ_complex[:, :rx * 2, :]
        # radar_data_4rx = IQ_complex[:, rx * 2:, :]
        radar_data_8rx = np.concatenate([IQ_complex[:, :rx, :], IQ_complex[:, rx * 2:, :]], axis=1)
        radar_data_4rx = IQ_complex[:, rx: rx * 2, :]
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
            
    def angle_fft(self, slow_time_samples, axis=1, shift=True):
        # Using cupy to perform fft
        # slow_time_samples: [num_range_bins, num_antenna, num_dopper_bins]
        angle_samples = np.fft.fft(slow_time_samples, n=self.num_angle_bins, axis=axis)
        if shift: 
            angle_samples = np.fft.fftshift(angle_samples, axes=axis)
        return angle_samples
    
    def elevation_fft(self, slow_time_samples, axis=2, shift=True):
        # Using cupy to perform fft
        elevation_samples = np.fft.fft(slow_time_samples, n=self.num_elevation_bins, axis=axis)
        if shift: 
            elevation_samples = np.fft.fftshift(elevation_samples, axes=axis)
        return elevation_samples
    
    def remove_direct_component(self, input_val, axis=-1):
        mean_val = np.mean(input_val, axis=axis)
        out_val = input_val - np.expand_dims(mean_val, axis=axis)
        return out_val
        
    def perform_beamforming(self, slow_time_samples, update_gain_mat=False):
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
        radar_data_8rx = self.remove_direct_component(radar_data_8rx, axis=0)
        radar_data_4rx = self.remove_direct_component(radar_data_4rx, axis=0)
        radar_data_8rx = self.range_fft(radar_data_8rx)
        radar_data_4rx = self.range_fft(radar_data_4rx)
        # radar_data_8rx = self.remove_direct_component(radar_data_8rx, axis=-1)
        # radar_data_4rx = self.remove_direct_component(radar_data_4rx, axis=-1)
        # Get doppler data
        radar_data_8rx = self.doppler_fft(radar_data_8rx, shift=False)
        radar_data_4rx = self.doppler_fft(radar_data_4rx, shift=False)
        # Padding to align: [range, azimuth, elevation, doppler]
        radar_data_4rx = np.pad(radar_data_4rx, ((0, 0), (2, 2), (0, 0)))
        radar_data = np.stack([radar_data_8rx, radar_data_4rx], axis=2) 
        radar_data = np.pad(radar_data, ((0, 0), (0, 0), (0, self.num_elevation_bins - 2), (0, 0)))
        # Get elevation data (along specific antenna)
        radar_data[:, 2: 6,:, :] = self.elevation_fft(radar_data[:, 2: 6,:, :], axis=-2, shift=False)
        # Get angle data
        radar_data = self.angle_fft(radar_data, shift=False)
        # Shift the fft result
        radar_data = np.fft.fftshift(radar_data, axes=(1, 2, 3))
        # Get the specific range
        # radar_data = radar_data[0: 64, :, :, :]
        radar_data = radar_data[100: 36: -1, :, :, :]
        # Select specific velocity
        radar_data = radar_data[:, :, :, self.num_doppler_bins // 2 - 8: self.num_doppler_bins // 2 + 8]
        # Flip at angle axis
        # radar_data = np.flip(radar_data, axis=0)
        
        return radar_data
