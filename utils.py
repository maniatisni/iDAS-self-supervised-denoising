from scipy.signal import tukey, butter, sosfiltfilt
import torch.nn as nn
import torch
from torch.nn.modules import padding
import torch.functional as F
import random
import numpy as np



def _butter_bandpass(lowcut, highcut, fs, order):

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        if low < 0:
            Wn = high
            btype = "lowpass"
        elif high < 0:
            Wn = low
            btype = "highpass"
        else:
            Wn = [low, high]
            btype = "bandpass"

        sos = butter(order, Wn, btype=btype, output="sos")
        return sos

def taper_filter(arr, fmin, fmax, samp_DAS, order=2):

    window_time = tukey(arr.shape[-1], 0.1)
    arr_wind = arr * window_time

    sos = _butter_bandpass(fmin, fmax, samp_DAS, order)
    arr_filt = sosfiltfilt(sos, arr_wind, axis=-1)

    return arr_filt



def masking(data,Nsub):
    Nch, Nt = data.shape

    masks = np.ones((Nch, Nsub, Nt, 1))
    eval_samples = np.zeros_like(masks)

    gutter = Nsub // 2
    mid = Nsub // 2

    for i in range(gutter):
        masks[i, i] = 0
        eval_samples[i, :, :, 0] = data[:Nsub]

    for i in range(gutter, Nch - gutter):
        start = i - mid
        stop = i + mid if Nsub % 2 == 0 else i+mid+1
        masks[i, mid] = 0
        eval_samples[i, :, :, 0] = data[start:stop]

    for i in range(Nch - gutter, Nch):
        masks[i, i - Nch] = 0
        eval_samples[i, :, :, 0] = data[-Nsub:]
    return eval_samples, masks