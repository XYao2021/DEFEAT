import torch.nn as nn
import torch.nn.functional as F
import torch

class SVHNModel(nn.Module):
    def __init__(self):
        super(SVHNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces 32x32 → 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Correct dimension calculation
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # conv1
        x = self.pool(self.relu(self.conv2(x)))  # pool after conv2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
