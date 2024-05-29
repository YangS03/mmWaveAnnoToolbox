import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR/preprocessing')

import os
import cv2
import yaml
import time
import cupy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mmRadar import FMCWRadar


class PreProcessor:
    def __init__(self):
        
        self.source_dir = r"/root/raw_data/"
        self.source_seqs = [
            # "2024-05-26-22-42-03",  # 60s
            # "2024-05-26-22-39-35",  # 30s
            # "2024-05-28-20-36-52",  # 5s
            "2024-05-28-21-29-20",  # 5s
        ]
        self.target_dir = r"/root/proc_data/HuPR/"
        
    def run_processing(self): 
        for seq_name in self.source_seqs:
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, path_video = self.load_folder(source_path_folder=os.path.join(self.source_dir, seq_name), load_video=True)        
            self.radar = FMCWRadar(mmwave_cfg)
            self.process_data(path_bin_hori, path_bin_vert, target_path_folder=os.path.join(self.target_dir, seq_name))
            # self.process_video(path_video, target_path_folder=os.path.join(self.target_dir, seq_name))
    
    def load_folder(self, source_path_folder, load_video=False): 
        print(os.path.join(source_path_folder, "radar_config.yaml"))
        with open(os.path.join(source_path_folder, "radar_config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        path_bin_hori = os.path.join(source_path_folder, "adc_data_hori.bin")
        path_bin_vert = os.path.join(source_path_folder, "adc_data_vert.bin")
        # path_video = os.path.join(source_path_folder, "record.mkv") if load_video else None
        path_video = os.path.join(source_path_folder, "video.mp4") if load_video else None
        
        return mmwave_cfg, path_bin_hori, path_bin_vert, path_video
    
    def process_video(self, path_video, target_path_folder): 
        cap = cv2.VideoCapture(path_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps // 10)
        
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            for _ in range(interval): 
                ret, frame = cap.read() 
            if ret: 
                folder_dir = os.path.join(target_path_folder, 'camera')
                os.makedirs(folder_dir, exist_ok=True)
                cv2.imwrite(os.path.join(folder_dir, "%09d.jpg" % idx_frame), frame)
    
    def process_data(self, path_bin_hori, path_bin_vert, target_path_folder=None): 
        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            data_frame_hori = data_complex_hori[idx_frame]
            cube_frame_hori = self.process_data_frame(data_frame_hori)
            self.save_data(cube_frame_hori, 'hori', idx_frame=idx_frame, target_dir=target_path_folder)
            data_frame_vert = data_complex_vert[idx_frame]
            cube_frame_vert = self.process_data_frame(data_frame_vert)
            self.save_data(cube_frame_vert, 'vert', idx_frame=idx_frame, target_dir=target_path_folder)
        
    def process_data_frame(self, data_frame):
        data_8rx, data_4rx = self.radar.parse_data(data_frame)
        data_cube = self.radar.get_RAED_data(data_8rx, data_4rx)    # [range, azimuth, elevation, doppler]
        return data_cube

    def save_data(self, data, view='hori', idx_frame=0, target_dir=None): 
        assert view in ['hori', 'vert'], 'Wrong view!!!'
        if isinstance(data, cp.ndarray): 
            data = data.get()   # cupy -> numpy
        folder_dir = os.path.join(target_dir, view)
        os.makedirs(folder_dir, exist_ok=True)
        np.save(os.path.join(folder_dir, "%09d.npy" % idx_frame), data)


if __name__ == '__main__': 
    # an example for processing data
    processor = PreProcessor()
    processor.run_processing()