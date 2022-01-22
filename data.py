import torch
import random
import numpy as np
from utils import *
from torch.utils.data import Dataset,Sampler
from glob import glob
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import os



###############################################
        # PyTorch Custom Masking DataSet #
###############################################
class mask_dataset(Dataset):
    """Distributed Acoustic Sensing dataset."""

    def __init__(self, data_path,
                f_min, f_max,N_sub, sampleRate=50,mode = 'train',transform=True):

        self.data_path = data_path
        self.filenames = glob(os.path.join(data_path, '*.npy'))
        self.f_min = f_min
        self.f_max = f_max
        self.sampleRate = sampleRate
        self.transform = transform
        self.N_sub = N_sub
        self.mode = mode

        # Load Data:
        self.eval_samples = []
        self.masks = []
        for file in self.filenames:
            data = np.load(file)
            if self.transform:
                data = taper_filter(data, self.f_min, self.f_max, self.sampleRate)
            x, y = self._masking(data)
            self.eval_samples.append(x)
            self.masks.append(y)

        self.eval_samples = np.vstack(self.eval_samples)
        self.masks = np.vstack(self.masks)

    def _masking(self,x):
        Nch = x.shape[0]
        Nt = x.shape[1]

        masks = np.ones((Nch, self.N_sub, Nt,1))
        eval_samples = np.zeros_like(masks)

        gutter = self.N_sub // 2
        mid = self.N_sub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :,0] = x[:self.N_sub]

        for i in range(gutter, Nch - gutter):
            start = i - mid
            stop = i + mid if self.N_sub % 2 == 0 else i + mid + 1
            masks[i, mid] = 0
            eval_samples[i, :, :,0] = x[start:stop]


        for i in range(Nch - gutter, Nch):
            masks[i, i - Nch] = 0
            eval_samples[i, :, :,0] = x[-self.N_sub:]
        
        return eval_samples, masks

    def __getitem__(self, idx):
        eval_samples = self.eval_samples[idx]
        masks = self.masks[idx]        
        masks = torch.tensor(masks.astype(np.float32).copy())
        eval_samples = torch.tensor(eval_samples.astype(np.float32).copy())
        return eval_samples, masks
    
    def __len__(self):
        return self.eval_samples.shape[0]



"""
 This is a tensorflow dataloader from original paper:

        "A Self-Supervised Deep Learning Approach for
        Blind Denoising and Waveform Coherence
        Enhancement in Distributed Acoustic Sensing data"
        by Martijn van den Ende et. al. 
    I just replaced the tensorflow module with the torch.utils.data.Dataset one.
"""
# rng = np.random.default_rng(42)

# class weird_DataLoader(torch.utils.data.Dataset):

#     def __init__(self, X, batch_size=16, batch_multiplier=10):
        
#         # Data matrix
#         self.X = X
#         # Number of samples
#         self.N_samples = X.shape[0]
#         # Number of stations
#         self.Nx = X.shape[1]
#         # Number of time sampling points
#         self.Nt = X.shape[2]
#         # Number of stations per batch sample
#         self.N_sub = 16
#         # Starting indices of the slices
#         self.station_inds = np.arange(self.Nx - self.N_sub)
#         # Batch size
#         self.batch_size = batch_size
#         self.batch_multiplier = batch_multiplier

#         self.on_epoch_end()

#     def __len__(self):
#         """ Number of mini-batches per epoch """
#         return int(self.batch_multiplier * self.N_samples * self.Nx / float(self.batch_size * self.N_sub))

#     def on_epoch_end(self):
#         """ Modify data """
#         self.__data_generation()
#         pass

#     def __getitem__(self, idx):
#         """ Select a mini-batch """
#         batch_size = self.batch_size
#         selection = slice(idx * batch_size, (idx + 1) * batch_size)
#         samples = np.expand_dims(self.samples[selection], -1)
#         masked_samples = np.expand_dims(self.masked_samples[selection], -1)
#         masks = np.expand_dims(self.masks[selection], -1)
#         return (samples, masks), masked_samples

#     def __data_generation(self):
#         """ Generate a total batch """
        
#         # Number of mini-batches
#         N_batch = self.__len__()
#         N_total = N_batch * self.batch_size
#         # Buffer for mini-batches
#         samples = np.zeros((N_total, self.N_sub, self.Nt))
#         # Buffer for masks
#         masks = np.ones_like(samples)
        
#         batch_inds = np.arange(N_total)
#         np.random.shuffle(batch_inds)
        
#         # Number of subsamples to create
#         n_mini = N_total // self.N_samples
        
#         # Loop over samples
#         for s, sample in enumerate(self.X):
#             # Random selection of station indices
#             selection = rng.choice(self.station_inds, size=n_mini, replace=False)
#             # Time reversal
#             order = rng.integers(low=0, high=2) * 2 - 1
#             sign = rng.integers(low=0, high=2) * 2 - 1
#             # Loop over station indices
#             for k, station in enumerate(selection):
#                 # Selection of stations
#                 station_slice = slice(station, station + self.N_sub)
#                 subsample = sign * sample[station_slice, ::order]
#                 # Get random index of this batch sample
#                 batch_ind = batch_inds[s * n_mini + k]
#                 # Store waveforms
#                 samples[batch_ind] = subsample
#                 # Select one waveform to blank
#                 blank_ind = rng.integers(low=0, high=self.N_sub)
#                 # Create mask
#                 masks[batch_ind, blank_ind] = 0
                
            
#         self.samples = samples
#         self.masks = masks
#         self.masked_samples = samples * (1 - masks)
#         pass

#     def generate_masks(self, samples):
#         """ Generate masks and masked samples """
#         N_masks = self.N_masks
#         N_patch = self.N_patch
#         Ny = samples.shape[2]
#         patch_inds = self.patch_inds
#         patch_radius = self.patch_radius
#         # Tile samples
#         samples = np.tile(samples, [N_masks, 1, 1])
#         # Add extra dimension
#         samples = np.expand_dims(samples, -1)
#         # Shuffle samples
#         inds = np.arange(samples.shape[0])
#         np.random.shuffle(inds)
#         samples = samples[inds]
#         # Generate complementary masks (patch = 1)
#         c_masks = np.zeros_like(samples)
#         for n in range(c_masks.shape[0]):
#             selection = rng.choice(patch_inds, size=N_patch, replace=False)
#             for sel in selection:
#                 i = sel // Ny
#                 j = sel % Ny
#                 slice_x = slice(i - patch_radius[0], i + patch_radius[0])
#                 slice_y = slice(j - patch_radius[1], j + patch_radius[1])
#                 c_masks[n, slice_x, slice_y] = 1
#         # Masks (patch = 0)
#         masks = 1 - c_masks
#         # Masked samples (for loss function)
#         masked_samples = c_masks * samples
#         return samples, masked_samples, masks

