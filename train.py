from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import torch

device = 'cpu'
log_interval = 250


class ChessValueDataset(Dataset):
    def __init__(self):
        data = np.load('processed/dataset_1k.npz')
        self.X = data['arr_0']
        self.Y = data['arr_1']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act='relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='reflect')
        self.act = nn.ReLU() if act == 'relu' else nn.Identity()
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.act(self.normalization(self.conv(x)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(5, 16, kernel_size=3, padding=1)
        self.conv1_rep = ConvBlock(16, 16, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1)
        self.conv2_rep = ConvBlock(32, 32, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3_rep = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv4_rep = ConvBlock(128, 128, kernel_size=3, padding=1)
        self.final = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # 8 x 8
        x = self.conv1(x)
        x = self.conv1_rep(x)
        x = self.conv1_rep(x)
        x = self.pool(x)

        # 4 x 4
        x = self.conv2(x)
        x = self.conv2_rep(x)
        x = self.conv2_rep(x)
        x = self.pool(x)

        # 2 x 2
        x = self.conv3(x)
        x = self.conv3_rep(x)
        x = self.conv3_rep(x)

        # 1 x 1
        x = self.conv4(x)
        x = self.conv4_rep(x)
        x = self.conv4_rep(x)
        x = self.pool(x)

        # 1 x 64
        x = x.view(-1, 128)
        x = self.final(x)

        return F.tanh(x.view((-1, )))


chess_dataset = ChessValueDataset()
model = Net()
train_loader = DataLoader(chess_dataset, batch_size=16)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

model.train()
for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
    data, target = data.type(torch.float32).permute((0, 3, 1, 2)), target.type(torch.float32)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
        print(f'\nBCE_Loss: {loss.item()}')
        torch.save(model.state_dict(), 'nets/value.pth.tar')
