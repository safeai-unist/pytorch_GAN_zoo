""" Style transfer network."""

import torch
from torch import nn, ops, Tensor

class Conv2d(nn.Module):
  
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, activation_fn=nn.ReLU(),
                 normalizer_fn=None, padding='same', **kwargs):
        super(Conv2d, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, **kwargs)
        self.bn = normalizer_fn
        self.activation_fn = activation_fn
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x, normalizer_fn, params, order = x
        x = self.conv(x)
        if normalizer_fn:
            x = normalizer_fn((x, params, order))
        if self.bn:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return (x, normalizer_fn, params, order + 1)

class Residual(nn.Module):
 
    def __init__(self, channels, kernel_size):
        super(Residual, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=kernel_size, stride=1)
        self.conv2 = Conv2d(channels, channels, kernel_size=kernel_size, stride=1, activation_fn=None)

    def forward(self, x):
        h_1 = self.conv1(x)
        h_2 = self.conv2(h_1)
        out1, _, _, _ = x
        out2, normalizer_fn, params, order = h_2
        return (out1 + out2, normalizer_fn, params, order)

class Upsampling(nn.Module):
 
    def __init__(self, stride, size, kernel_size, in_channels, out_channels, activation_fn=nn.ReLU()):
        super().__init__()
        self.stride = stride
        self.size = size
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, activation_fn=activation_fn)
    
    def forward(self, input_):
        x, normalizer_fn, params, order = input_
        _, _, height, width = x.shape
        x = nn.functional.interpolate(x, size=[height * self.stride, width * self.stride], mode='nearest')
        x = self.conv((x, normalizer_fn, params, order))
        return x

class Transform(nn.Module):
  
    def __init__(self, in_channels=3, alpha=1.0):
        super(Transform, self).__init__()
        self.contract = nn.Sequential(
            Conv2d(in_channels, int(alpha * 32), kernel_size=9, stride=1,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 32), eps=0.001)),
            Conv2d(int(alpha * 32), int(alpha * 64), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 64), eps=0.001)),
            Conv2d(int(alpha * 64), int(alpha * 128), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 128), eps=0.001))
        )
        self.residual = nn.Sequential(
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3)
            )
        self.expand = nn.Sequential(
            Upsampling(2, (32, 32), 3, int(alpha * 128), int(alpha * 64)),
            Upsampling(2, (64, 64), 3, int(alpha * 64), int(alpha * 32)),
            Upsampling(1, (128, 128), 9, int(alpha * 32), 3, activation_fn=nn.Sigmoid())
        )

    def forward(self, x):
        x, normalizer_fn, style_params = x
        out = self.contract((x, None, None, 0))
        x, _, _, _ = out
        x = self.residual((x, normalizer_fn, style_params, 0))
        out = self.expand(x)
        x, _, _, _ = out
        return x

class ConditionalStyleNorm(nn.Module):

    def __init__(self, style_params=None, activation_fn=None):
        super(ConditionalStyleNorm, self).__init__()
        self.style_params = style_params
        self.activation_fn = activation_fn

    def get_style_parameters(self, style_params):
        """Gets style normalization parameters."""
        var = []
        for i in style_params.keys():
            var.append(style_params[i])
            #var.append(style_params[i].unsqueeze(2).unsqueeze(3))
        return var

    def norm(self, x, mean, variance, style_parameters, variance_epsilon, order):
        """ Normalization function with specific parameters. """
        variance = variance + variance_epsilon
        inv = torch.rsqrt(variance)
        gamma = style_parameters[order*2+1]
        beta = style_parameters[order*2]
        if gamma is not None:
            inv = inv * gamma
        data1 = inv.to(x.dtype)
        data2 = x * data1
        data3 = mean * inv
        if gamma is not None:
            data4 = beta - data3
        else:
            data4 = -data3
        data5 = data2 + data4
        return data5

    def forward(self, input_):
        x, style_params, order = input_
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        variance = torch.var(x, dim=(2, 3), keepdim=True)
        
        style_parameters = self.get_style_parameters(style_params)
        output = self.norm(x, mean, variance, style_parameters, 1e-5, order)
        if self.activation_fn:
            output = self.activation_fn(output)
        return output
