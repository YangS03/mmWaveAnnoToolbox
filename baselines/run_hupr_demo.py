import sys
sys.path.append('.')
sys.path.append('./baselines/HuPR')

import os
import cv2
import yaml
import math
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

from modelInterface.model_interface import MInterface
from dataInterface.data_interface import DInterface

from HuPR.models import HuPRNet
from HuPR.datasets.dataset import HuPR3D_simple
from HuPR.misc import get_max_preds, generateTarget
from HuPR.datasets.base import Normalize
from HuPR.main import obj, parse_arg
from HuPR.misc.plot import plotHumanPoseOnly


if __name__ == "__main__":

    with open('./baselines/HuPR/config/mscsa_prgcn_demo.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    args = parse_arg()
    
    checkpoint = torch.load(os.path.join('./log/mscsa_prgcn/model_best.pth'))
    model = MInterface(HuPRNet, model_dict={'cfg': cfg}, model_state_dict=checkpoint['model_state_dict'])
    
    model.to('cuda')
    model.eval()
    
    data = DInterface(batch_size=1, dataset=HuPR3D_simple, dataset_dict={'cfg': cfg, 'args': args})
    data.setup(stage='test')
    
    print('Start testing...')
    for batch in tqdm(data.test_dataloader()): 
        VRDAEmap_hori = batch['VRDAEmap_hori'].cuda()
        VRDAEmap_vert = batch['VRDAEmap_vert'].cuda()
        heat_map, gcn_heat_map = model({'VRDAEmaps_hori': VRDAEmap_hori, 'VRDAEmaps_vert': VRDAEmap_vert})
        pred2d, _ = get_max_preds(gcn_heat_map.squeeze(1).detach().cpu().numpy())
        plotHumanPoseOnly(pred2d * 4, visDir='/root/viz', seqIdx=batch['seqIdx'], imageIdx=batch['imageId'])