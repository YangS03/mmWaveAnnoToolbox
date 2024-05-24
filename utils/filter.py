import cupy as np
from cupyx.scipy.signal import convolve2d, convolve

def average_filter_2d(matrix, kernel_size=3):
    
    filter_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    filtered_matrix = convolve2d(matrix, filter_kernel, mode='same', boundary='wrap')
    
    return filtered_matrix

def cfar_filter(signal, guard_band_size, training_band_size, threshold_factor, method="OS-CFAR", display=False):
    
    if signal.ndim == 1:
        if method == "OS-CFAR":
            return cfar_filter_1d_os(signal, guard_band_size, training_band_size, threshold_factor, display)
        elif method == "CA-CFAR":
            return cfar_filter_1d_ca(signal, guard_band_size, training_band_size, threshold_factor, display)

# OS-CFAR
def cfar_filter_2d(signal, guard_band_size_x, guard_band_size_y, refer_band_size_x, refer_band_size_y, threshold_factor):
    training_band_size_x = refer_band_size_x + guard_band_size_x
    training_band_size_y = refer_band_size_y + guard_band_size_y
    kernel = np.ones((2 * training_band_size_x + 1, 2 * training_band_size_y + 1))
    kernel[refer_band_size_x: -refer_band_size_x, refer_band_size_y: -refer_band_size_y] = 0
    kernel_size = (2 * training_band_size_x + 1) * (2 * training_band_size_y + 1) - \
        (2 * guard_band_size_x + 1) * (2 * guard_band_size_y + 1)
        
    conv_result = convolve2d(signal, kernel, mode='same', boundary='wrap') / kernel_size
        
    threshold = conv_result * threshold_factor
    detection_map = (signal > threshold).astype(np.int16)

    return detection_map 

# CA-CFAR
def cfar_filter_1d_ca(signal, guard_band_size, training_band_size, threshold_factor, display=False):

    kernel = np.ones(2 * training_band_size + 1)
    kernel[training_band_size - guard_band_size: \
        training_band_size + guard_band_size + 1] = 0
    n = 2 * (training_band_size - guard_band_size)

    conv_result = convolve(signal, kernel, mode='same')
    average_result = conv_result / n
    threshold = average_result * threshold_factor
    # threshold = np.percentile(conv_result, 100 * (1 - threshold_factor))
    
    detection_map = (signal > threshold).astype(int)
    detection_map[0: training_band_size] = 0
    detection_map[-training_band_size: ] = 0
    filter_result = signal * detection_map
    
    if display:
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(signal.get(), label="signal")
        plt.plot(threshold.get(), label="threshold")
        plt.legend()
        plt.show()
    
    return filter_result, detection_map

# OS-CFAR
def cfar_filter_1d_os(signal, guard_band_size, training_band_size, threshold_factor, display=False):

    kernel = np.ones(2 * training_band_size + 1)
    kernel[training_band_size - guard_band_size: \
        training_band_size + guard_band_size + 1] = 0
    n = 2 * (training_band_size - guard_band_size)

    conv_result = convolve(signal, kernel, mode='same')
    average_result = conv_result / n
    threshold = np.percentile(average_result, 100 * (1 - threshold_factor))
    
    detection_map = (signal > threshold).astype(int)
    filter_result = signal * detection_map
    
    if display:
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(signal.get(), label="signal")
        plt.plot(filter_result.get(), label="threshold")
        plt.legend()
        plt.show()
    
    return filter_result, detection_map
