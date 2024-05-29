import sys
sys.path.append('.')

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__': 
    # seq_name = 'single_1_8420'  # 60s
    # seq_name = '2024-05-28-21-29-20'  # 5s
    # seq_name = "2024-05-28-20-36-52"  # 5s
    # seq_name = "2024-05-26-22-39-35"  # 5s
    # seq_name = '2024-05-28-23-51-05-082852'
    seq_name = '2024-05-28-23-55-50-200299'
    data_dir = './data/HuPR/%s/hori' % seq_name
    data_list = sorted(os.listdir(data_dir))

    for data_name in tqdm(data_list): 
        data_hori = np.load(os.path.join(data_dir, data_name))   
        data_vert = np.load(os.path.join(data_dir.replace('hori', 'vert'), data_name))   

        plt.clf()
        plt.subplot(121)
        ramap = np.abs(data_hori).sum((2, 3))
        plt.imshow(ramap)
        plt.title('Range-Angle View')        
        plt.subplot(122)
        remap = np.abs(data_vert).sum((2, 3)).T
        plt.imshow(remap)
        plt.title('Range-Elevation View')        
        
        os.makedirs('./viz/%s/heatmap' % seq_name, exist_ok=True)
        plt.savefig('./viz/%s/heatmap/%s.png' % (seq_name, data_name[:9]))