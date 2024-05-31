""" Implementation of VGG-16 network for deriving content and style loss. """
from torch import nn, Tensor

class VGG(nn.Module):

    def __init__(self, in_channel=3):
        super(VGG, self).__init__()
        self.conv1 = self.make_layer(2, in_channel, 64, 3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = self.make_layer(2, 64, 128, 3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = self.make_layer(3, 128, 256, 3)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = self.make_layer(3, 256, 512, 3)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = self.make_layer(3, 512, 512, 3)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv6 = Conv2d(512, 4096, kernel_size=7, padding='valid')
        self.dropout1 = nn.Dropout(0.5)
        self.conv7 = Conv2d(4096, 4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.5)
        self.conv8 = Conv2d(4096, 1000, kernel_size=1, activation_fn=None)

    def make_layer(self, repeat, in_channel, out_channel, kernel_size):
        layer = []
        for _ in range(repeat):
            layer.append(Conv2d(in_channel, out_channel, kernel_size=kernel_size))
            in_channel = out_channel
        return nn.Sequential(layer)

    def forward(self, x):
        """ forward process """
        x *= 255.0
        _, _, height, width = x.shape
        cons = Tensor([123.68, 116.779, 103.939]).expand_dims(1).expand_dims(1)
        cons = cons.repeat(height, 1).repeat(width, 2).expand_dims(0)
        x -= cons

        end_points = {}
        x = self.conv1(x)
        end_points['vgg_16/conv1'] = x

        x = self.pool1(x)
        x = self.conv2(x)
        end_points['vgg_16/conv2'] = x

        x = self.pool2(x)
        x = self.conv3(x)
        end_points['vgg_16/conv3'] = x

        x = self.pool3(x)
        x = self.conv4(x)
        end_points['vgg_16/conv4'] = x

        x = self.pool4(x)
        x = self.conv5(x)
        end_points['vgg_16/conv5'] = x

        x = self.pool5(x)
        x = self.conv6(x)
        end_points['vgg_16/conv6'] = x

        x = self.dropout1(x)
        x = self.conv7(x)
        end_points['vgg_16/conv7'] = x

        x = self.dropout2(x)
        x = self.conv8(x)
        end_points['vgg_16/fc8'] = x

        return end_points

class Conv2d(nn.Module):
 
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1,
                 activation_fn=nn.ReLU(), padding='same', **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, has_bias=True, **kwargs)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
