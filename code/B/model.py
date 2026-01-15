# B/model.py
import torch
import torch.nn as nn

class SimpleCNN_B(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, capacity="base"):
        super().__init__()
        cap = {"small": 16, "base": 32, "large": 64}[capacity]
        c1, c2, c3 = cap, cap * 2, cap * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
