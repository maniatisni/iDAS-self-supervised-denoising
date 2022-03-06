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



# PyTorch Custom Masking DataSet #
class mask_dataset(Dataset):
    """Distributed Acoustic Sensing dataset."""

    def __init__(self, data_path,
                f_min, f_max,N_sub, sampleRate=50):

        self.data_path = data_path
        self.filenames = glob(os.path.join(data_path, '*.npy'))
        self.f_min = f_min
        self.f_max = f_max
        self.sampleRate = sampleRate
        self.N_sub = N_sub

        # Load Data:
        self.eval_samples = []
        self.masks = []
        for file in self.filenames:
            data = np.load(file)
            x, y = self._masking(data)
            self.eval_samples.append(x)
            self.masks.append(y)

        self.eval_samples = np.vstack(self.eval_samples)
        self.masks = np.vstack(self.masks)

    def _masking(self,x):
        Nch = x.shape[0]
        Nt = x.shape[1]

        masks = np.ones((Nch, self.N_sub, Nt))
        eval_samples = np.zeros_like(masks)

        gutter = self.N_sub // 2
        mid = self.N_sub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :] = x[:self.N_sub]

        for i in range(gutter, Nch - gutter):
            start = i - mid
            stop = i + mid if self.N_sub % 2 == 0 else i + mid + 1
            masks[i, mid] = 0
            eval_samples[i, :, :] = x[start:stop]


        for i in range(Nch - gutter, Nch):
            masks[i, i - Nch] = 0
            eval_samples[i, :, :] = x[-self.N_sub:]
        
        return eval_samples, masks

    def __getitem__(self, idx):
        eval_samples = self.eval_samples[idx]
        masks = self.masks[idx]        
        masks = torch.tensor(masks.astype(np.float32).copy())
        eval_samples = torch.tensor(eval_samples.astype(np.float32).copy())
        return eval_samples, masks
    
    def __len__(self):
        return self.eval_samples.shape[0]


# PyTorch Custom Masking DataSet for Synthetic Data - With Data Augmentation #
class synthetic_mask_dataset(Dataset):
    """Distributed Acoustic Sensing dataset."""

    def __init__(self, data_path,
                f_min, f_max,N_sub, sampleRate=50,mode = 'train'):

        self.data_path = data_path
        self.filenames = glob(os.path.join(data_path, '*.npy'))
        self.f_min = f_min
        self.f_max = f_max
        self.sampleRate = sampleRate
        self.N_sub = N_sub
        self.mode = mode

        # Load Data:
        self.eval_samples = []
        self.masks = []
        for file in self.filenames:
            data = np.load(file)
            x, y = self._masking(data)
            self.eval_samples.append(x)
            self.masks.append(y)

        self.eval_samples = np.vstack(self.eval_samples)
        self.masks = np.vstack(self.masks)

    def _masking(self,x):
        Nch = x.shape[0]
        Nt = x.shape[1]

        masks = np.ones((Nch, self.N_sub, Nt))
        eval_samples = np.zeros_like(masks)

        gutter = self.N_sub // 2
        mid = self.N_sub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :] = x[:self.N_sub]

        for i in range(gutter, Nch - gutter):
            start = i - mid
            stop = i + mid if self.N_sub % 2 == 0 else i + mid + 1
            masks[i, mid] = 0
            eval_samples[i, :, :] = x[start:stop]


        for i in range(Nch - gutter, Nch):
            masks[i, i - Nch] = 0
            eval_samples[i, :, :] = x[-self.N_sub:]
        
        return eval_samples, masks

    def __getitem__(self, idx):
        eval_samples = self.eval_samples[idx]
        masks = self.masks[idx]        
        masks = torch.tensor(masks.astype(np.float32).copy())
        eval_samples = torch.tensor(eval_samples.astype(np.float32).copy())
        rand_time = np.random.random()
        rand_polarity = np.random.random()
        # Time Reversal
        if rand_time > 0.5:
            eval_samples = eval_samples.flip(-1)
        # Polarity Flip
        if rand_polarity > 0.5:
            eval_samples = -eval_samples
        return eval_samples, masks
    
    def __len__(self):
        return self.eval_samples.shape[0]
