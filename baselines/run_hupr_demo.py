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

from HuPR.models import HuPRNet
from HuPR.misc import get_max_preds, generateTarget
from HuPR.datasets.base import Normalize, trans
from HuPR.main import obj


transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize()
            ])


def plotHumanPose(batch_joints, cfg=None, visDir=None, imageIdx=None, bbox=None, upsamplingSize=(256, 256), nrow=8, padding=2):
    """
    NO IMAGE DATA
    """
    for j in range(len(batch_joints)):
        namestr = '%09d'%imageIdx
        imageDir = os.path.join(visDir, 'single_%d'%int(namestr[:4]))
        if not os.path.isdir(imageDir):
            os.mkdir(imageDir)
        imagePath = os.path.join(imageDir, '%09d.png'%int(namestr[-4:]))
        s_joints = np.expand_dims(batch_joints[j], axis=0)
        batch_image = torch.zeros((1, 3, 256, 256))
        grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        ndarr = ndarr.copy()

        nmaps = batch_image.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height = int(batch_image.size(2) + padding)
        width = int(batch_image.size(3) + padding)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                joints = s_joints[k]
                for joint in joints:
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                k = k + 1
        joints_edges = [[(int(joints[0][0]), int(joints[0][1])), (int(joints[1][0]), int(joints[1][1]))],
                        [(int(joints[1][0]), int(joints[1][1])), (int(joints[2][0]), int(joints[2][1]))],
                        [(int(joints[0][0]), int(joints[0][1])), (int(joints[3][0]), int(joints[3][1]))],
                        [(int(joints[3][0]), int(joints[3][1])), (int(joints[4][0]), int(joints[4][1]))],
                        [(int(joints[4][0]), int(joints[4][1])), (int(joints[5][0]), int(joints[5][1]))],
                        [(int(joints[0][0]), int(joints[0][1])), (int(joints[6][0]), int(joints[6][1]))],
                        [(int(joints[3][0]), int(joints[3][1])), (int(joints[6][0]), int(joints[6][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[7][0]), int(joints[7][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[8][0]), int(joints[8][1]))],
                        [(int(joints[6][0]), int(joints[6][1])), (int(joints[11][0]), int(joints[11][1]))],
                        [(int(joints[8][0]), int(joints[8][1])), (int(joints[9][0]), int(joints[9][1]))],
                        [(int(joints[9][0]), int(joints[9][1])), (int(joints[10][0]), int(joints[10][1]))],
                        [(int(joints[11][0]), int(joints[11][1])), (int(joints[12][0]), int(joints[12][1]))],
                        [(int(joints[12][0]), int(joints[12][1])), (int(joints[13][0]), int(joints[13][1]))],
        ]
        for joint_edge in joints_edges:
            ndarr = cv2.line(ndarr, joint_edge[0], joint_edge[1], [255, 0, 0], 1)

        if bbox is not None:
            topleft = (int(bbox[j][0].item()), int(bbox[j][1].item()))
            topright = (int(bbox[j][0].item() + bbox[j][2].item()), int(bbox[j][1]))
            botleft = (int(bbox[j][0].item()), int(bbox[j][1].item() + bbox[j][3].item()))
            botright = (int(bbox[j][0].item() + bbox[j][2].item()), int(bbox[j][1].item() + bbox[j][3].item()))
            ndarr = cv2.line(ndarr, topleft, topright, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, topleft, botleft, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, topright, botright, [0, 255, 0], 1)
            ndarr = cv2.line(ndarr, botleft, botright, [0, 255, 0], 1)

        cv2.imwrite(imagePath, ndarr[:, :, [2, 1, 0]])


if __name__ == "__main__":

    with open('./baselines/HuPR/config/mscsa_prgcn.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    
    # simplified dataloader  
    index = 98
    duration = 100
    numGroupFrames = 8
    numChirps = 16
    numFrames = 8
    rangeSize = 64
    azimuthSize = 64
    elevationSize = 8
    padSize = index % duration
    idx = index - numGroupFrames//2 - 1
    
    VRDAEmaps_hori = torch.zeros((numGroupFrames, numFrames, 2, rangeSize, azimuthSize, elevationSize))
    VRDAEmaps_vert = torch.zeros((numGroupFrames, numFrames, 2, rangeSize, azimuthSize, elevationSize))
    VRDAERealImag_hori_frames = np.load("data/HuPR/test/data_hori.npy")
    VRDAERealImag_vert_frames = np.load("data/HuPR/test/data_vert.npy")
    
    frames = np.clip(range(index - numGroupFrames//2, index + numGroupFrames//2), 0, duration - 1)
    for idx, idx_frame in enumerate(frames):
    
        VRDAERealImag_hori = VRDAERealImag_hori_frames[idx_frame]
        VRDAERealImag_vert = VRDAERealImag_vert_frames[idx_frame]
        
        idxSampleChirps = 0
        for idxChirps in range(numChirps//2 - numFrames//2, numChirps//2 + numFrames//2):   # select the low-velocity chirps
            VRDAEmaps_hori[idx, idxSampleChirps, 1, :, :, :] = transformFunc(VRDAERealImag_hori[:, :, :, idxChirps].imag).permute(1, 2, 0)
            VRDAEmaps_hori[idx, idxSampleChirps, 0, :, :, :] = transformFunc(VRDAERealImag_hori[:, :, :, idxChirps].real).permute(1, 2, 0)
            VRDAEmaps_vert[idx, idxSampleChirps, 0, :, :, :] = transformFunc(VRDAERealImag_vert[:, :, :, idxChirps].real).permute(1, 2, 0)
            VRDAEmaps_vert[idx, idxSampleChirps, 1, :, :, :] = transformFunc(VRDAERealImag_vert[:, :, :, idxChirps].imag).permute(1, 2, 0)
            idxSampleChirps += 1
    
    # add batch
    VRDAEmaps_hori = VRDAEmaps_hori.unsqueeze(0).to('cuda')
    VRDAEmaps_vert = VRDAEmaps_vert.unsqueeze(0).to('cuda')
    
    checkpoint = torch.load(os.path.join('./log/mscsa_prgcn/model_best.pth'))
    model = HuPRNet(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()
    
    heat_map, gcn_heat_map = model(VRDAEmaps_hori, VRDAEmaps_vert)
    pred2d, _ = get_max_preds(gcn_heat_map.squeeze(1).detach().cpu().numpy())
    
    plotHumanPose(pred2d * 4, visDir='./viz', imageIdx=idx)