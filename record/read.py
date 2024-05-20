import numpy as np
import mmwave as mm
from mmwave.dataloader import DCA1000

import matplotlib.pyplot as plt

dca = DCA1000()

plt.ion()

while True: 
    adc_data = dca.read()
    adc_data = mm.dsp.clutter_removal(adc_data)
    data_mat = dca.organize(adc_data, 80, 8, 256)
    
    range_spec = mm.dsp.range_processing(data_mat)
    range_spec = np.abs(range_spec).mean(axis=(0, 1))
    
    
    plt.plot(range_spec)
    plt.pause(0.01)
    plt.clf()