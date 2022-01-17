#%%
from model import UNet
from data import DAS_Dataset
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_path = "C:\\Users\\nikos\\Desktop\\DAS-data\\various-earthquakes\\"

# %%

# GPU
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device is {device}.')
# Model Input & Output is: (batch_size, Nsub, 30000)
batch_size = 256
model = UNet(input_bands=batch_size,hidden_channels=12, output_classes=batch_size)
model.to(device)

# Mean Square Error 
criterion = nn.MSELoss()

# Optimizer:
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# epochs
num_epochs = 1
#%%
if __name__ == "__main__":
    # Initialize Dataset and DataLoader
    dataset = DAS_Dataset(data_path, f_min = 1, f_max = 20, N_sub = 32)
    data_loader = DataLoader(dataset, shuffle = False, batch_size=batch_size, num_workers=6,drop_last=True)
    epochs_losses = []
    batch_indexes = [i for i in range(batch_size)]
    ## TRAIN ##
    for epoch in range(num_epochs):
        print(f"Starting epoch: #{epoch}")
        batch_losses = []
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
            if len(batch_indexes) % 100 == 0:
                print(f"Loss is {loss.item()}")
    epochs_losses.append(np.mean(batch_losses))

    plt.plot(batch_losses)
    plt.yscale("log")
    plt.title(f"Batches losses for batch_size = {batch_size}")
    plt.savefig("batch_losses.png")
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, "C:\\Users\\nikos\\Desktop\\denoising\\model.pth")