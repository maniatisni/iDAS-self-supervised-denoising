# IMPORTS # 

from msilib.schema import Error
import numpy as np
from scipy.signal import resample
from utils import *
from glob import glob
import os

"""
The idea here is to take 3 files of 30 sec each @ 1kHz, so 90 sec in total or 90,000 samples,
concatenate them,
convert to strain rate,
keep only frequencies between 1 and 10 Hz,
downsample so we have 7168 (=7*1024) samples in the end,
split into train and test on channel axis,
standardize, 
and write into separate folders.

Final Sample Rate = 79.64 Hz ~ 80Hz.

"""

# Initial and Final Sample Rates
# These are true for Santorini DAS.
init_samplerate = 1000.
init_samples = 41000

# Modify these accordingly.
# f_final = N_samples_final * f_init / N_samples_init

final_samples = 2048
final_samplerate = final_samples*init_samplerate/init_samples
print(f"Final Sample Rate is {final_samplerate:.2f} Hz.")


# Number of DAS spatial channels


# Santorini DAS has 5568 channels, we choose the following range.
channel_min = 1700
channel_max = 2700


# File Paths # 
input_data_path = "E:\\October-19-20\\"
output_train_data_path = "C:\\Users\\nikos\\Desktop\\denoising\\train-events"
output_val_data_path = "C:\\Users\\nikos\\Desktop\\denoising\\validation-events"
filenames = glob(os.path.join(input_data_path, '*.npy'))
file_index = 942

name = f"event_{filenames[file_index].split('UTC_')[-1]}"


print("Reading data...")
# Downsampling and saving as .npy files
d1 = np.load(filenames[file_index])[channel_min:channel_max]
d2 = np.load(filenames[file_index+1])[channel_min:channel_max]
d = np.concatenate((d1,d2), axis = 1)[:,8000:8000+41000]


if d.shape[1] != init_samples:
    print("Check dimensions again")
    exit()

# Convert to strain rate
d = (116*init_samplerate*d)/81920

# Butterworth filter, keep only frequencies from 1 to 10 Hz.
d = taper_filter(d,1,10,init_samplerate)

print("Downsampling...")
# Actual Downsampling.
dr = np.zeros((d.shape[0],final_samples))
for channel in range(0,d.shape[0]):
        dr[channel] = resample(d[channel,], num=final_samples)

#split into train-test
dr_train, dr_val = dr[:800,:], dr[800:,:]

# Normalize, divide by standard dev.
dr_train = dr_train/dr_train.std()
dr_val = dr_val/dr_val.std()

print("Saving...")
np.save(os.path.join(output_train_data_path,name), dr_train)
print(f"saved train file, shape: {dr_train.shape}")
np.save(os.path.join(output_val_data_path,name), dr_val)
print(f"saved val file, shape: {dr_val.shape}")