import matplotlib.pyplot as plt
import os
from numpy import fmin
from model import UNet
from data import DAS_Dataset,RamKillerDataset
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import perf_counter
from glob import glob

def train():
    data_path = "C:\\Users\\nikos\\Desktop\\DAS-data\\various-earthquakes\\"
    filenames = glob(os.path.join(data_path, '*.npy'))
    print(f"{len(filenames)} files in total. ")
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}.')

    # Model Input & Output is: (batch_size, Nsub, 30000)
    batch_size = 64
    model = UNet(input_bands=batch_size,hidden_channels=12, output_classes=batch_size)
    model.to(device)

    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    num_epochs = 10
    epochs_losses = []
    batch_indexes = [i for i in range(batch_size)]
    ## TRAIN ##
    for epoch in range(num_epochs):
        print(f"Starting epoch: #{epoch}")
        batch_losses = []
        for file in filenames:
            start = perf_counter()
            ds = RamKillerDataset(file, f_min=1,f_max = 10, N_sub= 32)
            data_loader = DataLoader(ds, batch_size=batch_size, num_workers=6,drop_last=True, shuffle = True, prefetch_factor=2)
            for row, x, y in tqdm(data_loader):
                x = x.to(device)
                y = y.to(device)
                row = row.to(device)

                rows_inds = [int(i) for i in row]
                optimizer.zero_grad()

                output = model(x[None,...]).squeeze()

                #Blank out all channels except target channel
                output = output[batch_indexes, rows_inds,:]
                loss = criterion(y,output)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            end = perf_counter()
            print(f"File trained in {end-start:.2f} sec, Loss is: {loss.item()}.")
        print(f"Finished epoch: #{epoch}.")
        epochs_losses.append(np.mean(batch_losses))
    plt.plot(epochs_losses)
    plt.yscale("log")
    plt.title(f"Epoch losses - batch_size={batch_size}")
    plt.savefig("10epochs.png")
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, "C:\\Users\\nikos\\Desktop\\denoising\\model.pth")
#%%
if __name__ == "__main__":
    train()