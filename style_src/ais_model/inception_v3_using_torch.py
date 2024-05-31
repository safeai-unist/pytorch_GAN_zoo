"""
Implementation of Inception_v3 network for encodering style images.
"""

from torch import nn
from ais_model.inception_block_using_torch import InceptionA, InceptionB, InceptionC, Conv2d

class InceptionV3(nn.Module):

    def __init__(self, in_channels=3):
        super(InceptionV3, self).__init__()

        self.model = nn.Sequential(
            # 299 * 299 * 3
            Conv2d(in_channels, out_channels=32, kernel_size=3, stride=2, padding='valid'),
            nn.ReLU(),
            # 149 * 149 * 32
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='valid'),
            nn.ReLU(),
            # 147 * 147 * 32
            
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            # 147 * 147 * 64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 73 * 73 * 64
            Conv2d(in_channels=64, out_channels=80, kernel_size=1, padding='valid'),
            nn.ReLU(),
            # 73 * 73 * 80
            Conv2d(in_channels=80, out_channels=192, kernel_size=3, padding='valid'),
            nn.ReLU(),
            # 71 * 71 * 192
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 35 * 35 * 192
            InceptionA(192, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=32),
            # 35 * 35 * 256   5b
            InceptionA(256, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=64),
            # 35 * 35 * 288   5c
            InceptionA(288, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=64),
            # 35 * 35 * 288   5d
            InceptionB(288, c1=384, c2=[64, 96, 96]),
            # 17 * 17 * 768   6a
            InceptionC(768, c1=192, c2=[128, 128, 192], c3=[128, 128, 128, 128, 192], c4=192),
            # 17 * 17 * 768   6b
            InceptionC(768, c1=192, c2=[160, 160, 192], c3=[160, 160, 160, 160, 192], c4=192),
            # 17 * 17 * 768   6c
            InceptionC(768, c1=192, c2=[160, 160, 192], c3=[160, 160, 160, 160, 192], c4=192),
            # 17 * 17 * 768   6d
            InceptionC(768, c1=192, c2=[192, 192, 192], c3=[192, 192, 192, 192, 192], c4=192),
        )

    def forward(self, x):
        x = self.model(x)
        return x
