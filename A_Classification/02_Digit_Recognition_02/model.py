# Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
# https://github.com/nordengt/LeNet5-Implementation

import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)

        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.s2(self.tanh(x))
        x = self.c3(x)
        x = self.s4(self.tanh(x))
        x = self.c5(x)
        x = self.f6(self.flatten(x))
        x = self.output(self.tanh(x))
        return x
    
class LeNet5Modern(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=5)
        self.fc1 = nn.Linear(4*4*feature_channels, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x