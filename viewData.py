import numpy as np
import matplotlib.pyplot as plt
import mpld3

# with open('/root/raw_data/hupr/single_1/hori/adc_data.bin') as f: 
with open('/root/raw_data/demo/2024-05-30-15-16-03-529798/adc_data_hori.bin') as f: 
    raw_data = np.fromfile(f, dtype=np.int16)


file_size = raw_data.shape[0] // 2

data = raw_data
# data1 = data[1::4].copy()
# data2 = data[2::4].copy()
# data[1::4] = data2
# data[2::4] = data1
lvds0 = data[0::2]
lvds1 = data[1::2]
lvds0 = lvds0.reshape(-1, 64, 12, 256)
lvds1 = lvds1.reshape(-1, 64, 12, 256)

sig_data = lvds1 + 1j * lvds0
# sig_data = lvds0 + 1j * lvds1

mean_val = sig_data.mean(axis=1)
sig_data -= np.expand_dims(mean_val, axis=1)

plt.plot(sig_data.real[0, 32, 0, :])
plt.plot(sig_data.imag[0, 32, 0, :])
mpld3.show()

sig = sig_data[0, 0, 0, :]
fft = np.fft.fft(sig)

plt.plot(np.abs(fft))
mpld3.show()
