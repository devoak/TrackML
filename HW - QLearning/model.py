import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class QModel(nn.Module):

    def __init__(self):
        super(QModel, self).__init__()
        self.nl = nn.ELU()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.lin = nn.Linear(56 * 56, 4)

    def forward(self, x):
        x = self.nl(self.bn1(self.conv1(x)))
        x = self.nl(self.bn2(self.conv2(x)))
        x = self.nl(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
