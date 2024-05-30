import numpy as np

with open('/root/raw_data/demo/2024-05-29-23-42-19-849302/adc_data_vert.bin') as f: 
    data = np.fromfile(f, dtype=np.int16)
print(data.shape)