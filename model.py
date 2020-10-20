import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

class ConvNet(nn.Module):
    def __init__(self, num_class=6):
        super().__init__()
        layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.cnn = nn.Sequential(
            layer1,
            layer2,
            layer3,
            Flatten(),
            nn.Linear(64*28*28, 1000),
            nn.Linear(1000, num_class)
        )

    def forward(self, x):
        out = self.cnn(x)
        
        return out

        

if __name__ == '__main__':
    input_test = torch.ones(10, 3, 224, 224)
    net = ConvNet()
    loss = net(input_test)
    print(loss)	
        
  
