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


def to_numpy(data): 
    if isinstance(data, cp.ndarray): 
        return data.get()
    else: 
        return data


class PreProcessor:
    def __init__(self):
        
        self.source_dir = r"/root/raw_data/demo/"
        self.source_seqs = [
            # '2024-05-29-22-22-05-443181',   # left
            # '2024-05-29-22-21-29-074018',   # right
            # '2024-05-29-22-22-37-027792',    # T
            # '2024-05-29-23-38-57-931262',   # L
            # '2024-05-29-23-40-00-290270',   # R
            # '2024-05-29-23-41-25-579382',   # L far
            '2024-05-29-23-42-19-849302',   # R far
            # '2024-05-29-23-42-58-051479',   # T
        ]
        self.target_dir = r"/root/proc_data/HuPR/"
        
    def run_processing(self): 
        for seq_name in self.source_seqs:
            print('Processing', seq_name)
            mmwave_cfg, path_bin_hori, path_bin_vert, path_video = self.load_folder(source_path_folder=os.path.join(self.source_dir, seq_name), load_video=True)        
            self.radar = FMCWRadar(mmwave_cfg)
            self.process_data(path_bin_hori, path_bin_vert, target_path_folder=os.path.join(self.target_dir, seq_name), seq_name=seq_name)
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
    
    def process_data(self, path_bin_hori, path_bin_vert, target_path_folder=None, seq_name=None): 
        data_complex_hori = self.radar.read_data(path_bin_hori, complex=True)
        data_complex_vert = self.radar.read_data(path_bin_vert, complex=True)
        for idx_frame in tqdm(range(self.radar.num_frames)): 
            data_frame_hori = data_complex_hori[idx_frame]
            data_frame_vert = data_complex_vert[idx_frame]
            cube_frame_hori = to_numpy(self.process_data_frame(data_frame_hori))
            cube_frame_vert = to_numpy(self.process_data_frame(data_frame_vert))
            self.save_data(cube_frame_hori, 'hori', idx_frame=idx_frame, target_dir=target_path_folder)
            self.save_data(cube_frame_vert, 'vert', idx_frame=idx_frame, target_dir=target_path_folder)
            self.save_plot(cube_frame_hori, cube_frame_vert, idx_frame=idx_frame, seq_name=seq_name)
        
    def process_data_frame(self, data_frame):
        data_8rx, data_4rx = self.radar.parse_data(data_frame)
        data_cube = self.radar.get_RAED_data(data_8rx, data_4rx)    # [range, azimuth, elevation, doppler]
        return data_cube
    
    def save_plot(self, data_hori, data_vert, idx_frame=0, seq_name=None): 
        plt.clf()
        plt.subplot(121)
        ramap = np.abs(data_hori).sum((0, 3))
        plt.imshow(ramap)
        plt.title('Range-Angle View')        
        plt.subplot(122)
        remap = np.abs(data_vert).sum((0, 3)).T
        plt.imshow(remap)
        plt.title('Range-Elevation View')  
        if not os.path.exists('/root/viz/%s/heatmap' % seq_name): 
            os.makedirs('/root/viz/%s/heatmap' % seq_name)
        plt.savefig('/root/viz/%s/heatmap/%09d.png' % (seq_name, idx_frame))

    def save_data(self, data, view='hori', idx_frame=0, target_dir=None): 
        assert view in ['hori', 'vert'], 'Wrong view!!!'
        folder_dir = os.path.join(target_dir, view)
        os.makedirs(folder_dir, exist_ok=True)
        np.save(os.path.join(folder_dir, "%09d.npy" % idx_frame), data)


if __name__ == '__main__': 
    # an example for processing data
    processor = PreProcessor()
    processor.run_processing()