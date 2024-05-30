import sys
sys.path.append('.')

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__': 
    seq_name = '2024-05-29-23-42-19-849302' # R far
    
    data_dir = './data/HuPR/%s/hori' % seq_name 
    data_list = sorted(os.listdir(data_dir))

    for data_name in tqdm(data_list): 
        data_hori = np.load(os.path.join(data_dir, data_name))   
        data_vert = np.load(os.path.join(data_dir.replace('hori', 'vert'), data_name))   

        plt.clf()
        plt.subplot(121)
        ramap = np.abs(data_hori).sum((0, 3))
        plt.imshow(ramap)
        plt.title('Range-Angle View')        
        plt.subplot(122)
        remap = np.abs(data_vert).sum((0, 3)).T
        plt.imshow(remap)
        plt.title('Range-Elevation View')        
        
        os.makedirs('./viz/%s/heatmap' % seq_name, exist_ok=True)
        plt.savefig('./viz/%s/heatmap/%s.png' % (seq_name, data_name[:9]))