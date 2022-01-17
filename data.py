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
        # PyTorch Custom DataSet #
###############################################

# class DAS_Dataset(Dataset):
#     """Distributed Acoustic Sensing dataset."""

#     def __init__(self, data_path,
#                 f_min, f_max,N_sub, channel_min = 1700, channel_max = 2300, 
#                 sampleRate=1000,transform=True):
#         """
#         Args:
#         data_path: Path for the folder containing the DAS data.
#         f_min: Minimum Frequency for the butterworth filter. Cuts frequencies below that value.
#         f_max: Maximum Frequency for the butterworth filter. Cuts frequencies above that value.
#         sampleRate: Sampling Rate of the data OR how many samples per second are taken. Default is 1000 (Hz).
#         channel_min: channel to start training from.
#         channel_max: channell to end training to.
#         N_sub: Number of channels per batch

#         """
#         self.data_path = data_path
#         self.filenames = [x for x in os.listdir(data_path) if x.endswith(".npy")]

#         self.f_min = f_min
#         self.f_max = f_max
#         self.sampleRate = sampleRate
#         self.transform = transform
#         self.channel_min = channel_min
#         self.channel_max = channel_max
#         self.N_sub = N_sub

#         # mapping 
#         self.all_indexes = []
#         for file in self.filenames:
#             for row in range(self.channel_max-self.channel_min):
#                 self.all_indexes.append((file,row))       

#     def __getitem__(self, idx):
#         file, row = self.all_indexes[idx]
#         x = np.load(f"{self.data_path}{file}", mmap_mode='r')[self.channel_min:self.channel_max]
#         low_index = int(row - self.N_sub/2)
#         high_index = int(row + self.N_sub/2)
        
#         # Normalization, this causes minimal data leakage
#         x = x/x.std()
#         # Copy because assigning values to slices messes things up
#         y_ = x[row].copy()
#         # x_ = x.copy()
#         # Zero out the target channel, the model will predict this.
#         x[row] = 0
        
#         # if target is close to zero, then pick range [0, N_sub], target is not centered.
#         if int(row - self.N_sub/2) <= 0:
#             low_index = 0
#             high_index = self.N_sub
#         # if target is close to max channel, pick range [Nsub, max_channel], target is not centered again.
#         if int(row + self.N_sub/2) >= self.channel_max-self.channel_min:
#             high_index = self.channel_max-self.channel_min
#             low_index = (self.channel_max - self.channel_min) - self.N_sub

#         x_ = x[low_index:high_index]
#         # Keep only frequencies from f_min to f_max.
#         if self.transform:
#             x_ = taper_filter(x_, self.f_min, self.f_max, self.sampleRate)
#             y_ = taper_filter(y_, self.f_min, self.f_max, self.sampleRate)

#         # print(self.all_indexes[idx])
#         # Necessary types
#         new_row = np.where(np.sum(np.abs(x_), axis=1)==0)[0]
#         x_ = torch.tensor(x_.astype(np.float32).copy())
#         y_ = torch.tensor(y_.astype(np.float32).copy())
#         return new_row, x_, y_
    
#     def __len__(self):
#         return len(self.all_indexes)

# ###############################################
#         # PyTorch Custom RAM Killer DataSet #
# ###############################################


# class RamKillerDataset(Dataset):
#     """Distributed Acoustic Sensing dataset."""

#     def __init__(self, data_path,
#                 f_min, f_max,N_sub,sampleRate=1000,channel_min = 1700,channel_max = 2364,transform=True):
#         self.channel_min = channel_min
#         self.channel_max = channel_max
#         self.data = np.load(data_path,mmap_mode='r')[channel_min:channel_max]
#         self.data = (116*sampleRate*self.data)/81920
#         self.data = self.data/self.data.std()
#         self.data = taper_filter(self.data, f_min, f_max, sampleRate)
#         self.f_min = f_min
#         self.f_max = f_max
#         self.sampleRate = sampleRate
#         self.transform = transform
#         self.N_sub = N_sub

#     def __getitem__(self, row):
#         low_index = int(row - self.N_sub/2)
#         high_index = int(row + self.N_sub/2)
#         # Copy because assigning values to slices messes things up
#         y_ = self.data[row].copy()
#         # if target is close to zero, then pick range [0, N_sub], target is not centered.
#         if int(row - self.N_sub/2) <= 0:
#             low_index = 0
#             high_index = self.N_sub
#         # if target is close to max channel, pick range [Nsub, max_channel], target is not centered again.
#         if int(row + self.N_sub/2) >= self.data.shape[0]:
#             high_index = self.data.shape[0]
#             low_index = self.data.shape[0]-self.N_sub

#         x_ = self.data[low_index:high_index].copy()
#         # Keep only frequencies from f_min to f_max.
#         # if self.transform:
#         #     x_ = taper_filter(x_, self.f_min, self.f_max, self.sampleRate)
#         #     y_ = taper_filter(y_, self.f_min, self.f_max, self.sampleRate)
#         # Necessary types
#         new_row = int(np.where(np.all(y_==x_, axis = 1))[0])
#         x_[new_row] = 0
#         x_ = torch.tensor(x_.astype(np.float32).copy())
#         y_ = torch.tensor(y_.astype(np.float32).copy())
#         return new_row, x_, y_
    
#     def __len__(self):
#         return len(self.data)


###############################################
        # PyTorch Custom Mask DataSet #
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
        # self.eval_samples = np.stack(self.eval_samples, axis = 1)
        # self.masks = np.stack(self.masks, axis = 1)
        # N_ch = self.eval_samples.shape[0]
        # N_files = self.eval_samples.shape[1]
        # N_samples = self.eval_samples.shape[3]

        # self.eval_samples = self.eval_samples.reshape(N_ch*N_files,self.N_sub,N_samples)
        # self.masks = self.masks.reshape(N_ch*N_files,self.N_sub,N_samples)

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
        # eval_samples with mask applied meaning the target channels are blanked.
        
        masks = torch.tensor(masks.astype(np.float32).copy())
        eval_samples = torch.tensor(eval_samples.astype(np.float32).copy())
        return eval_samples, masks
    
    def __len__(self):
        return self.eval_samples.shape[0]




rng = np.random.default_rng(42)

class weird_DataLoader(torch.utils.data.Dataset):

    def __init__(self, X, batch_size=16, batch_multiplier=10):
        
        # Data matrix
        self.X = X
        # Number of samples
        self.N_samples = X.shape[0]
        # Number of stations
        self.Nx = X.shape[1]
        # Number of time sampling points
        self.Nt = X.shape[2]
        # Number of stations per batch sample
        self.N_sub = 16
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - self.N_sub)
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier

        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.N_samples * self.Nx / float(self.batch_size * self.N_sub))

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        pass

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
                station_slice = slice(station, station + self.N_sub)
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
        pass

    def generate_masks(self, samples):
        """ Generate masks and masked samples """
        N_masks = self.N_masks
        N_patch = self.N_patch
        Ny = samples.shape[2]
        patch_inds = self.patch_inds
        patch_radius = self.patch_radius
        # Tile samples
        samples = np.tile(samples, [N_masks, 1, 1])
        # Add extra dimension
        samples = np.expand_dims(samples, -1)
        # Shuffle samples
        inds = np.arange(samples.shape[0])
        np.random.shuffle(inds)
        samples = samples[inds]
        # Generate complementary masks (patch = 1)
        c_masks = np.zeros_like(samples)
        for n in range(c_masks.shape[0]):
            selection = rng.choice(patch_inds, size=N_patch, replace=False)
            for sel in selection:
                i = sel // Ny
                j = sel % Ny
                slice_x = slice(i - patch_radius[0], i + patch_radius[0])
                slice_y = slice(j - patch_radius[1], j + patch_radius[1])
                c_masks[n, slice_x, slice_y] = 1
        # Masks (patch = 0)
        masks = 1 - c_masks
        # Masked samples (for loss function)
        masked_samples = c_masks * samples
        return samples, masked_samples, masks
# class chunk_dataset(Dataset):
#     """Distributed Acoustic Sensing dataset."""

#     def __init__(self, data_path,
#                 f_min, f_max,N_sub, channel_min = 1700, channel_max = 2300, 
#                 sampleRate=1000,transform=True):
#         """
#         Args:
#         data_path: Path for the folder containing the DAS data.
#         f_min: Minimum Frequency for the butterworth filter. Cuts frequencies below that value.
#         f_max: Maximum Frequency for the butterworth filter. Cuts frequencies above that value.
#         sampleRate: Sampling Rate of the data OR how many samples per second are taken. Default is 1000 (Hz).
#         channel_min: channel to start training from.
#         channel_max: channell to end training to.
#         N_sub: Number of channels per batch

#         """
#         self.data_path = data_path
#         self.filenames = [x for x in os.listdir(data_path) if x.endswith(".npy")]

#         self.f_min = f_min
#         self.f_max = f_max
#         self.sampleRate = sampleRate
#         self.transform = transform
#         self.channel_min = channel_min
#         self.channel_max = channel_max
#         self.N_sub = N_sub
      

#     def __getitem__(self, idx):
#         file = self.filenames[idx]
#         x = np.load(f"{self.data_path}{file}", mmap_mode='r')[self.channel_min:self.channel_max]
#         chunks_x_list = []
#         chunks_y_list = []
#         for row in range(x.shape[0]):
#             low_index = int(row - self.N_sub/2)
#             high_index = int(row + self.N_sub/2)
#             # if target is close to zero, then pick range [0, N_sub], target is not centered.
#             if int(row - self.N_sub/2) <= 0:
#                 low_index = 0
#                 high_index = self.N_sub
#             # if target is close to max channel, pick range [Nsub, max_channel], target is not centered again.
#             if int(row + self.N_sub/2) >= self.channel_max-self.channel_min:
#                 high_index = self.channel_max-self.channel_min
#                 low_index = (self.channel_max - self.channel_min) - self.N_sub

#             # Normalization, this causes minimal data leakage
#             x = x/x.std()
#             # Copy because assigning values to slices messes things up
#             y_ = x[row].copy()
#             # x_ = x.copy()
#             # Zero out the target channel, the model will predict this.
#             x[row] = 0

#             x_ = x[low_index:high_index]
#             # Keep only frequencies from f_min to f_max.
#             if self.transform:
#                 x_ = taper_filter(x_, self.f_min, self.f_max, self.sampleRate)
#                 y_ = taper_filter(y_, self.f_min, self.f_max, self.sampleRate)
#             chunks_x_list.append(x_)
#             chunks_y_list.append(y_)

#         chunks_x = np.stack(chunks_x_list)
#         chunks_y = np.stack(chunks_y_list)
#         # print(self.all_indexes[idx])
#         # Necessary types
#         chunks_x = torch.tensor(chunks_x.astype(np.float32).copy())
#         chunks_y = torch.tensor(chunks_y.astype(np.float32).copy())
#         return chunks_x, chunks_y
    
#     def __len__(self):
#         return len(self.filenames)

# class MyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, data_path, N_sub, batch_size, channel_min = 1700, channel_max = 2300):
#         super(MyIterableDataset).__init__()
#         self.data_path = data_path
#         self.filenames = [x for x in os.listdir(data_path) if x.endswith(".npy")]
#         self.channel_min = channel_min
#         self.channel_max = channel_max
#         self.N_sub = N_sub
#         self.batch_size = batch_size

#     def file_read(self,file):
#         data = np.load(f"{self.data_path}{file}",mmap_mode='r')[self.channel_min:self.channel_max]
#         return data
#     def sliding_window(self,data,row):
#         low_index = int(row - self.N_sub/2)
#         high_index = int(row + self.N_sub/2)
#         # if target is close to zero, then pick range [0, N_sub], target is not centered.
#         if int(row - self.N_sub/2) <= 0:
#             low_index = 0
#             high_index = self.N_sub
#         # if target is close to max channel, pick range [Nsub, max_channel], target is not centered again.
#         if int(row + self.N_sub/2) >= self.channel_max-self.channel_min:
#             high_index = self.channel_max-self.channel_min
#             low_index = (self.channel_max - self.channel_min) - self.N_sub

#         # Normalization, this causes minimal data leakage
#         data = data/data.std()
#         # Copy because assigning values to slices messes things up
#         y_ = data[row].copy()

#         # Zero out the target channel, the model will predict this.
#         data[row] = 0
#         x_ = data[low_index:high_index]
#         # Keep only frequencies from f_min to f_max.
#         # x_ = taper_filter(x_, self.f_min, self.f_max, self.sampleRate)
#         # y_ = taper_filter(y_, self.f_min, self.f_max, self.sampleRate)
#         new_row = np.where(np.sum(np.abs(x_), axis=1)==0)[0]
#         x_ = torch.tensor(x_.astype(np.float32).copy())
#         y_ = torch.tensor(y_.astype(np.float32).copy())

#         return new_row, x_, y_
        
#     def __iter__(self):
#         for file in self.filenames:
#             data = self.file_read(file)
#             num_rows = list(range(data.shape[0]))
#             start, end = 0, len(num_rows)
#             worker_info = torch.utils.data.get_worker_info()
#             if worker_info is None:  # single-process data loading, return the full iterator
#                 iter_start = start
#                 iter_end = end
#             else: 
#                 per_worker = int(len(num_rows) // float(worker_info.num_workers))
#                 worker_id = worker_info.id
#                 iter_start = start + worker_id * per_worker
#                 iter_end = min(iter_start + per_worker, end)       
                 
#             yield list(map(lambda x: self.sliding_window(data,x), iter(num_rows[iter_start:iter_end])))

#     def __len__(self):
#         return len(self.filenames)

# class IterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, data_path, N_sub, channel_min, channel_max, batch_size):
#         self.batch_size = batch_size
#         self.data_path = data_path
#         self.filenames = [x for x in os.listdir(data_path) if x.endswith(".npy")]
#         self.channel_min = channel_min
#         self.channel_max = channel_max
#         self.N_sub = N_sub
#         self.batch_size = batch_size
    
#     def file_read(self,file):
#         data = np.load(f"{self.data_path}{file}",mmap_mode='r')[self.channel_min:self.channel_max]
#         return data
#     def sliding_window(self,data,row):
#         low_index = int(row - self.N_sub/2)
#         high_index = int(row + self.N_sub/2)
#         # if target is close to zero, then pick range [0, N_sub], target is not centered.
#         if int(row - self.N_sub/2) <= 0:
#             low_index = 0
#             high_index = self.N_sub
#         # if target is close to max channel, pick range [Nsub, max_channel], target is not centered again.
#         if int(row + self.N_sub/2) >= self.channel_max-self.channel_min:
#             high_index = self.channel_max-self.channel_min
#             low_index = (self.channel_max - self.channel_min) - self.N_sub

#         # Normalization, this causes minimal data leakage
#         data = data/data.std()
#         # Copy because assigning values to slices messes things up
#         y_ = data[row].copy()

#         # Zero out the target channel, the model will predict this.
#         data[row] = 0
#         x_ = data[low_index:high_index]
#         # Keep only frequencies from f_min to f_max.
#         # x_ = taper_filter(x_, self.f_min, self.f_max, self.sampleRate)
#         # y_ = taper_filter(y_, self.f_min, self.f_max, self.sampleRate)
#         new_row = np.where(np.sum(np.abs(x_), axis=1)==0)[0]
#         x_ = torch.tensor(x_.astype(np.float32).copy())
#         y_ = torch.tensor(y_.astype(np.float32).copy())

#         return new_row, x_, y_
#     def __iter__(self):
#         for file in self.filenames:
#             data = self.file_read(file)
#             num_rows = data.shape[0]
#             for row in range(num_rows):
#                 yield self.sliding_window(data, row)
#     @classmethod
#     def split_datasets(cls, data_path, N_sub, channel_min, channel_max, batch_size, max_workers):
#         super.__init__()
#         for n in range(max_workers, 0, -1):
#             if batch_size % n == 0:
#                 num_workers = n
#                 break
#         split_size = batch_size // num_workers

#         return [cls(data_path, batch_size=split_size) for _ in range(num_workers)]

# class MultiStreamDataLoader:
#     def __init__(self, datasets):
#         self.datasets = datasets
    
#     def get_stream_loaders(self):
#         return zip(*[DataLoader(dataset, num_workers=0, batch_size=None) for dataset in self.datasets])

#     def __iter__(self):
#         for batch_parts in self.get_stream_loaders():
#             yield list(chain(*batch_parts))