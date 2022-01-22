from scipy.signal import tukey, butter, sosfiltfilt
import numpy as np


#########################################
# BUTTERWORTH FILTER #
#########################################
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

###########################################
# Mean Local Waveform coherence CC #
###########################################

def xcorr(x, y):
    
    # FFT of x and conjugation
    X_bar = np.fft.rfft(x).conj()
    Y = np.fft.rfft(y)
    
    # Compute norm of data
    norm_x_sq = np.sum(x**2)
    norm_y_sq = np.sum(y**2)
    norm = np.sqrt(norm_x_sq * norm_y_sq)
    
    # Correlation coefficients
    R = np.fft.irfft(X_bar * Y) / norm
    
    # Return correlation coefficient
    return np.max(R)

def compute_xcorr_window(x):
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan
    
    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])
    
    return np.nanmean(Cxy)

def compute_moving_coherence(data, bin_size):
    
    N_ch = data.shape[0]
    
    cc = np.zeros(N_ch)
    
    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])
        
    return cc