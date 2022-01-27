import numpy as np
from scipy.signal import resample
from utils import *
from glob import glob
import os


# These are true for Santorini DAS.
init_samplerate = 1000.
init_samples = 30000

# Modify these accordingly.
# f_final = N_samples_final * f_init / N_samples_init

final_samples = 2048
final_samplerate = final_samples*init_samplerate/init_samples
print(f"Final Sample Rate is {final_samplerate:.2f} Hz.")


# Number of DAS spatial channels
# Santorini DAS has 5568 channels, we choose the following range.
channel_min = 1700
channel_max = 2300


# File Paths # 
input_data_path = "C:\\Users\\nikos\\Desktop\\denoising\\test-full"
output_data_path = "test"
filenames = glob(os.path.join(input_data_path, '*.npy'))
print(f"{len(filenames)} files will be downsampled.")


# Downsampling and saving as .npy files
for file in filenames:
        d = np.load(file)[channel_min:channel_max]
        # Convert to strain rate
        d = (116*init_samplerate*d)/81920
        # Butterworth filter, keep only frequencies from 1 to 10 Hz.
        d = taper_filter(d,1,10,init_samplerate)
        # Actual Downsampling.
        dr = np.zeros((d.shape[0],final_samples))
        for channel in range(0,d.shape[0]):
                dr[channel] = resample(d[channel,], num=final_samples)

        # Normalize, divide by standard dev.
        dr = dr/dr.std()
        np.save(os.path.join(output_data_path,file.split("\\")[-1]), dr)
