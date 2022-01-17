from utils import *
from model import UNet
from data import mask_dataset_batch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter
from glob import glob 
import os

def main():
    ds = mask_dataset_batch(data_path="train", f_min = 1, f_max = 10, N_sub=32, batch_size = 56)
    loader = DataLoader(ds, batch_size=56, num_workers = 6,shuffle = False)
    for i,(x,y) in tqdm(enumerate(loader)):
        pass
if __name__ == "__main__":
    main()

