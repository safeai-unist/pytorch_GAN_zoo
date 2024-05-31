"""Several inception blocks of Inception_v3 network."""
import torch
from torch import nn, ops

class InceptionA(nn.Module):

    def __init__(self, num_channels: int, c1: int, c2, c3, c4: int):
        super(InceptionA, self).__init__()
        self.conv1 = Conv2d(num_channels, c1, kernel_size=1, padding='same')
        self.conv2 = nn.Sequential(
            Conv2d(num_channels, c2[0], kernel_size=1, padding='same'),
            Conv2d(c2[0], c2[1], kernel_size=5, padding='same')
        )
        self.conv3 = nn.Sequential(
            Conv2d(num_channels, c3[0], kernel_size=1, padding='same'),
            Conv2d(c3[0], c3[1], kernel_size=3, padding='same'),
            Conv2d(c3[1], c3[2], kernel_size=3, padding='same')
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(num_channels, c4, kernel_size=1, padding='same')
        )

    def forward(self, x):
        # print("A1::::", self.conv1(x).size())
        # print("A2::::", self.conv2(x).size())
        # print("A3::::", self.conv3(x).size())
        # print("A4::::", self.conv4(x).size())
        return torch.cat((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)), dim=1)

class InceptionB(nn.Module):

    def __init__(self, num_channels: int, c1: int, c2):
        super(InceptionB, self).__init__()
        self.conv1 = Conv2d(num_channels, c1, kernel_size=3, stride=2, padding='valid')
        self.conv2 = nn.Sequential(
            Conv2d(num_channels, c2[0], kernel_size=1),
            Conv2d(c2[0], c2[1], kernel_size=3, padding='same'),
            Conv2d(c2[1], c2[2], kernel_size=3, stride=2, padding='valid'),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        # print("B1::::", self.conv1(x).size())
        # print("B2::::", self.conv2(x).size())
        # print("B3::::", self.conv3(x).size())

        return torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), dim=1)

class InceptionC(nn.Module):

    def __init__(self, num_channels: int, c1: int, c2, c3, c4: int):
        super(InceptionC, self).__init__()
        self.conv1 = Conv2d(num_channels, c1, kernel_size=1, padding='same')
        self.conv2 = nn.Sequential(
            Conv2d(num_channels, c2[0], kernel_size=1, padding='same'),
            Conv2d(c2[0], c2[1], kernel_size=(1, 7), padding='same'),
            Conv2d(c2[1], c2[2], kernel_size=(7, 1), padding='same'),
        )
        self.conv3 = nn.Sequential(
            Conv2d(num_channels, c3[0], kernel_size=1, padding='same'),
            Conv2d(c3[0], c3[1], kernel_size=(7, 1), padding='same'),
            Conv2d(c3[1], c3[2], kernel_size=(1, 7), padding='same'),
            Conv2d(c3[2], c3[3], kernel_size=(7, 1), padding='same'),
            Conv2d(c3[3], c3[4], kernel_size=(1, 7), padding='same')
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(num_channels, c4, kernel_size=1, padding='same')
        )

    def forward(self, x):
        # print("C1::::", self.conv1(x).size())
        # print("C2::::", self.conv2(x).size())
        # print("C3::::", self.conv3(x).size())
        # print("C4::::", self.conv3(x).size())
        return torch.cat((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)), dim=1)

class Conv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.activation_fn = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        return x
