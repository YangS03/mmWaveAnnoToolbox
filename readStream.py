import time
import cupy as cp
import numpy as np
import mmwave as mm
import open3d as o3d
from mmwave.dataloader import DCA1000
from radar.fmcw_radar import FMCWRadar
from pcGenerator.pc_generator import get_pointcloud
import radar_configs.mmwave_radar.iwr1843 as cfg

if __name__ == "__main__": 
    dca = DCA1000(static_ip='192.168.33.31', adc_ip='192.168.33.181', data_port=4088, config_port=4086)
    radar = FMCWRadar(cfg)
    
    vis = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    while True: 
        adc_data = dca.read()
        data_frame = dca.organize(adc_data, cfg.mmwave.num_slow_samples, cfg.mmwave.num_antenna, cfg.mmwave.num_fast_samples)
        
        tic = time.time()
        pointcloud = get_pointcloud(data_frame).T
        pointcloud = pointcloud.get()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        # update point cloud dynamically
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        print('Time taken: ', time.time() - tic)
        
        