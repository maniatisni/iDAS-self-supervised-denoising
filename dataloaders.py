from utils import taper_filter
from torch.utils.data import Dataset
import numpy as np
import random as python_random

# Python random seed
seed = 42
python_random.seed(seed)
# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)




class synthetic_mask_dataset(Dataset):

    """
    Distributed Acoustic Sensing dataset for synthetic data.
    Contains data generation to simulate DAS recordings and
    data augmentation (polarity flips, time reversals).
    """

    def __init__(self, X, Nt=2048, N_sub=10, batch_size=16, batch_multiplier=10):

        # Data matrix
        self.X = X
        # Number of stations
        self.Nx = X.shape[0]
        # Number of time sampling points in data
        self.Nt_all = X.shape[1]
        # Number of time sampling points in a slice
        self.Nt = Nt
        # Number of stations per batch sample
        self.N_sub = N_sub
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier
        # Number of mini-batches
        N_batch = self.__len__()
        N_total = N_batch * self.batch_size
        Nt = self.Nt
        N_sub = self.N_sub
        # Buffer for mini-batches
        samples = np.zeros((N_total, N_sub, Nt))
        # Buffer for masks
        masks = np.ones_like(samples)
        
        N_mid = self.Nt_all // 2
        t_starts = rng.integers(low=N_mid-Nt, high=N_mid-Nt//2, size=N_total)
        
        X = self.X
        
        s_min = 1/10_000.
        s_max = 1/200.
        
        gauge = 19.2
        samp = 50.
        
        log_SNR_min = -2
        log_SNR_max = 4
        
        
        # Loop over samples
        for s, t_start in enumerate(t_starts):
            
            sample_ind = rng.integers(low=0, high=self.Nx)            
            t_slice = slice(t_start, t_start + Nt)
            
            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            # Polarity flip
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Move-out direction
            direction = rng.integers(low=0, high=2) * 2 - 1
            
            slowness = rng.random() * (s_max - s_min) + s_min
            shift = direction * gauge * slowness * samp
            
            sample = sign * X[sample_ind, ::order]
            
            SNR = rng.random() * (log_SNR_max - log_SNR_min) + log_SNR_min
            SNR = 10**(0.5 * SNR)
            amp = 2 * SNR / np.abs(sample).max()
            sample = sample * amp
            
            for i in range(N_sub):                
                samples[s, i] = np.roll(sample, int(i*shift))[t_slice]
            
            # Select one waveform to blank
            blank_ind = rng.integers(low=0, high=self.N_sub)
            masks[s, blank_ind] = 0
            
        gutter = 100
        noise = rng.standard_normal((N_total * N_sub, Nt + 2*gutter))
        noise = taper_filter(noise, fmin=1.0, fmax=10.0, samp_DAS=samp)[:, gutter:-gutter]
        noise = noise.reshape(*samples.shape)
        
        noisy_samples = samples + noise
        for s, sample in enumerate(noisy_samples):
            noisy_samples[s] = sample / sample.std()
            
        self.samples = noisy_samples
        self.masks = masks
        self.masked_samples = noisy_samples * (1 - masks)
    
    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.Nx * self.Nt_all / float(self.batch_size * self.Nt))
    
    def __getitem__(self,idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        samples = np.expand_dims(self.samples[selection], -1)
        masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        masks = np.expand_dims(self.masks[selection], -1)
        return (samples, masks), masked_samples


class DAS_data_generator(Dataset):
    """
    This is the PyTorch Dataset for real DAS data.
    It masks the data the way needed for the j-invariant reconstructions,
    and also augments the data (polarity flips, time reversals).
    """

    def __init__(self, X, N_sub=11, batch_size=32, batch_multiplier=10):
        
        # Data matrix
        self.X = X
        # Number of samples
        self.N_samples = X.shape[0]
        # Number of stations
        self.Nx = X.shape[1]
        # Number of time sampling points
        self.Nt = X.shape[2]
        # Number of stations per batch sample
        self.N_sub = N_sub
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - N_sub)
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier
        self.__data_generation()

    def __len__(self):
        """ Number of mini-batches per epoch """
        # return self.batch_multiplier * self.N_samples * int(self.Nx / float(self.batch_size * self.N_sub))
        return self.batch_multiplier * self.N_samples

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        samples = np.expand_dims(self.samples[selection], -1)
        masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        masks = np.expand_dims(self.masks[selection], -1)
        return (samples, masks), masked_samples

    def __data_generation(self):
        """ Generate a total batch """
        
        # Number of mini-batches
        N_batch = self.__len__()
        N_total = N_batch * self.batch_size
        # Buffer for mini-batches
        samples = np.zeros((N_total, self.N_sub, self.Nt))
        # Buffer for masks
        masks = np.ones_like(samples)
        
        batch_inds = np.arange(N_total)
        np.random.shuffle(batch_inds)
        
        # Number of subsamples to create
        n_mini = N_total // self.N_samples
        
        # Loop over samples
        for s, sample in enumerate(self.X):
            # Random selection of station indices
            selection = rng.choice(self.station_inds, size=n_mini, replace=False)
            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Loop over station indices
            for k, station in enumerate(selection):
                # Selection of stations
                station_slice = slice(k, k + self.N_sub)
                subsample = sign * sample[station_slice, ::order]
                # Get random index of this batch sample
                batch_ind = batch_inds[s * n_mini + k]
                # Store waveforms
                samples[batch_ind] = subsample
                # Select one waveform to blank
                blank_ind = rng.integers(low=0, high=self.N_sub)
                # Create mask
                masks[batch_ind, blank_ind] = 0
                
        self.samples = samples
        self.masks = masks
        self.masked_samples = samples * (1 - masks)