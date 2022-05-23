# %%
# IMPORTS # 

from msilib.schema import Error
import numpy as np
from scipy.signal import resample
from utils import *
from glob import glob
import os

"""
The idea here is to take 2 files of 30 sec each @ 1kHz, keep only 41 sec 41,000 samples,
concatenate them,
convert to strain rate,
keep only frequencies between 1 and 10 Hz,
downsample so we have 2048 samples in the end,
standardize, 
and write into a new file.

Final Sample Rate = 49.95 ~ 50 Hz.
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

for i in range(11,12):
    # File Paths # 
    input_data_path = f"E:\\events-raw\\{i}"
    output_data_path = "..\\data\\santorini-DAS-2048-v2\\"
    # Hard Coded - Which events to read from each file and where to start.
    if i==1:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[:2]
        start = 4000
    elif i==2:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[:2]
        start = 9000
    elif i==3:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[:2]
        start = 0
    elif i==4:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 25000
    elif i==5:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 26000
    elif i==6:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 24000
    elif i==7:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 7000
    elif i==8:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 12000
    elif i==9:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 20000
    elif i==10:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 14000
    elif i==11:
        # filenames = glob(os.path.join(input_data_path, '*.npy'))[:3]
        # start = 20000
        filenames = glob(os.path.join(input_data_path, '*.npy'))[3:]
        start = 10000
    elif i==12:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 0
    elif i==13:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 27000
    elif i==14:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[:3]
        start = 21000
    elif i==15:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[:3]
        start = 12000
    elif i==16:
        filenames = glob(os.path.join(input_data_path, '*.npy'))[1:4]
        start = 0


    print(f"i == {i}")
    name = filenames[0].split("\\")[-1]
    print("Reading data...")
    d = np.concatenate([np.load(file) for file in filenames],axis=1)
    d = d[channel_min:channel_max,start:start+init_samples]

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

    print("Saving...")
    np.save(os.path.join(output_data_path, f"event_{name}"),dr)
    print(os.path.join(output_data_path, f"event_{name}"))
    # np.save(os.path.join(output_train_data_path,name), dr_train)
    # print(f"saved train file, shape: {dr_train.shape}")
    # np.save(os.path.join(output_val_data_path,name), dr_val)
    # print(f"saved val file, shape: {dr_val.shape}")
    # %%
