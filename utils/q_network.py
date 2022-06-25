import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_shape, action_size):
        super(QNetwork, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=state_shape[-1], out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.globmaxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        out = torch.permute(x, (0,3,1,2))  # put channels first
        out = F.relu(self.conv1(out))
        # out = self.maxpool(out)
        out = F.relu(self.conv2(out))
        # out = self.maxpool(out)
        out = F.relu(self.conv3(out))
        out = self.globmaxpool(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
