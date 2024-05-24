import sys
sys.path.append('.')
import time
import cupy as np
import open3d as o3d
from einops import rearrange, repeat
from matplotlib import pyplot as plt

from radar.fmcw_radar import FMCWRadar 
from radar.beamformer import CaponBeamformer, MUSICBeamformer
import radar_configs.mmwave_radar.iwr1843 as cfg


def preprocess_data(radar, data_frame, top_size):
    # perform range fft
    range_result = radar.range_fft(data_frame)
    range_result = radar.remove_direct_component(range_result)
    # perform doppler fft
    doppler_result = radar.doppler_fft(range_result)
    doppler_result_db = doppler_result.sum(axis=1)
    doppler_result_db = np.abs(doppler_result_db)
    doppler_result_db = np.log10(np.abs(doppler_result_db))
    # filter out the bins which are too close or too far from radar
    doppler_result_db[:8, :] = 0
    doppler_result_db[-8:, :] = 0
    # filter out the bins lower than energy threshold
    filter_result = np.zeros_like(doppler_result_db)    # [num_range_bins, num_doppler_bins]
    energy_thre = np.sort(doppler_result_db.ravel())[-top_size - 1]
    filter_result[doppler_result_db > energy_thre] = True  
    # get range-doppler indices
    det_peaks_indices = np.argwhere(filter_result == True)
    range_scale = det_peaks_indices[:, 0].astype(np.float64) * cfg.mmwave.range_resolution
    veloc_scale = (det_peaks_indices[:, 1] - cfg.mmwave.num_doppler_bins // 2).astype(np.float64) * cfg.mmwave.doppler_resolution
    # get aoa inputs (doppler value at top samples)
    energy_result = doppler_result_db[filter_result == True]
    # azimuth and elevation estimation
    doppler_result = rearrange(doppler_result, 'r a d -> a r d')
    aoa_input = doppler_result[:, filter_result == True] 
    return aoa_input, doppler_result, energy_result, range_scale, veloc_scale


def get_pointcloud(radar, data_frame, top_size=64, method='fft'): 
    aoa_input, _, energy_result, range_scale, veloc_scale = preprocess_data(radar, data_frame, top_size)
    # split value
    num_rx = cfg.mmwave.num_rx
    azimuth_ant = aoa_input[0: num_rx * 2, :]
    elevation_ant = aoa_input[num_rx * 2: , :]
    x_vec, y_vec, z_vec = naive_xyz(azimuth_ant, elevation_ant)
    x, y, z = x_vec * range_scale, y_vec * range_scale, z_vec * range_scale
    point_cloud = np.concatenate((x, y, z, veloc_scale, energy_result, range_scale))
    point_cloud = np.reshape(point_cloud, (6, -1))
    point_cloud = point_cloud[:, y_vec != 0]
    return point_cloud


def find_peak(data_org, peak_source=None):
    if peak_source is None: 
        peak_source = data_org
    data_max = np.argmax(np.abs(data_org), axis=0)
    data_peak = np.zeros_like(data_max, dtype=np.complex_) 
    for i in range(len(data_max)):
        data_peak[i] = peak_source[data_max[i], i]
    return data_max, data_peak


def compute_phase_shift(data_ant, method='music'):
    if method == 'fft': 
        data_fft = np.fft.fft(data_ant, n=cfg.mmwave.num_angle_bins, axis=0)
        data_org = np.fft.fftshift(data_fft, axes=0)
        data_max, data_peak = find_peak(data_org)
    elif method == 'beamform': 
        beamformer = CaponBeamformer(num_steps=cfg.mmwave.num_angle_bins, num_antennas=data_ant.shape[0])
        data_ant_ = data_ant.T
        _, bm_weight = beamformer.steering(data_ant_[:, :, np.newaxis])
        data_org = np.einsum('i j k, i k -> i j', np.conj(bm_weight).transpose(0, 2, 1), data_ant_).T
        data_max, data_peak = find_peak(data_org)
    elif method == 'music': 
        data_fft = np.fft.fft(data_ant, n=cfg.mmwave.num_angle_bins, axis=0)
        data_fft = np.fft.fftshift(data_fft, axes=0)
        beamformer = MUSICBeamformer(num_steps=cfg.mmwave.num_angle_bins, num_antennas=data_ant.shape[0])
        data_ant_ = data_ant.T      
        power_spectrum = np.zeros_like(data_fft)
        for idx in range(data_ant_.shape[0]):  
            power_spectrum[:, idx] = beamformer.steering(data_ant_[idx, :, np.newaxis])
        data_max, data_peak = find_peak(power_spectrum, peak_source=data_fft)
    return data_max, data_peak


def naive_xyz(azimuth_ant, elevation_ant): 
    # azimuth estimation
    azimuth_max, azimuth_peak = compute_phase_shift(azimuth_ant)
    wx = 2 * cfg.mmwave.angle_resolution * (azimuth_max - cfg.mmwave.num_angle_bins // 2)
    # elevation estimation
    _, elevation_peak = compute_phase_shift(elevation_ant)
    wz = np.angle(azimuth_peak * elevation_peak.conj() * np.exp(1j * 2 * wx))
    # get xyz coordinate
    x_vector = wx / np.pi
    z_vector = wz / np.pi    
    y_vector = 1 - x_vector ** 2 - z_vector ** 2
    x_vector[y_vector < 0] = 0
    z_vector[y_vector < 0] = 0
    y_vector[y_vector < 0] = 0
    y_vector = np.sqrt(y_vector)
    
    return x_vector, y_vector, z_vector


if __name__ == '__main__': 
    
    vis = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    radar = FMCWRadar(cfg)
    bin_filename = 'dca1000/saved_files/hori/adc_data_Raw_0.bin'
    bin_data = radar.read_data(bin_filename, complex=True)
    # for idx_frame in range(cfg.mmwave.num_frames):
    idx_global = 0
    while True: 
        tic = time.time()
        idx_frame = idx_global % cfg.mmwave.num_frames
        idx_global += 1
        data_frame = bin_data[idx_frame]    # [num_chirps, num_attennas, num_adc_samples]
        pointcloud = get_pointcloud(radar, data_frame, top_size=64, method='beamform').T
        pointcloud = pointcloud.get()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        color_map = repeat(pointcloud[:, 4], 'n -> n 3')
        color_map = 1 - color_map / color_map.max()
        pcd.colors = o3d.utility.Vector3dVector(color_map)
        # update point cloud dynamically
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)