import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR/preprocessing')

import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mmRadar import FMCWRadar


class PreProcessor:
    def __init__(self):
        
        self.source_dir = r"E:\CollectedData\wxg_test\2024-05-24-17-18-12"
        self.target_dir = r".\data\HuPR\test"

        mmwave_cfg, path_bin_vert, path_bin_hori = self.load_folder(self.source_dir)        
        self.radar = FMCWRadar(mmwave_cfg)
        self.process_data(path_bin_hori, path_bin_vert)
    
    def load_folder(self, source_path_folder): 
        with open(os.path.join(source_path_folder, "config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        path_bin_hori = os.path.join(source_path_folder, "adc_data_hori.bin")
        path_bin_vert = os.path.join(source_path_folder, "adc_data_vert.bin")
        
        return mmwave_cfg, path_bin_vert, path_bin_hori
    
    def process_data(self, path_bin_hori, path_bin_vert): 

        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            data_frame_hori = data_complex_hori[idx_frame]
            data_frame_vert = data_complex_vert[idx_frame]
    
            cube_frame_vert = self.process_data_frame(data_frame_hori)
            cube_frame_hori = self.process_data_frame(data_frame_vert)
            
            cube_complex_hori = cube_frame_vert.get()    # from cupy to numpy
            cube_complex_vert = cube_frame_hori.get()
            
            self.save_data(cube_complex_hori, cube_complex_vert, idx_frame)

    def process_data_frame(self, data_frame):
        data_8rx, data_4rx = self.radar.parse_data(data_frame)
        data_cube = self.radar.get_RAED_data(data_8rx, data_4rx)    # [range, azimuth, elevation, doppler]
        return data_cube

    def save_data(self, data_hori, data_vert, idx): 
        os.makedirs(os.path.join(self.target_dir, 'hori'), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, 'vert'), exist_ok=True)
        
        plt.ion()
        plt.clf()
        plt.imshow(np.sum(np.abs(data_hori), axis=(2, 3)))
        plt.pause(0.01)
        
        np.save(os.path.join(self.target_dir, 'hori', "%09d.npy" % idx), data_hori)
        np.save(os.path.join(self.target_dir, 'vert', "%09d.npy" % idx), data_vert)


if __name__ == '__main__': 
    # an example for processing data
    processor = PreProcessor()
    