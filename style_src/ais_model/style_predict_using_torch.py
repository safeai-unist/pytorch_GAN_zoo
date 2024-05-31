""" Style predict network."""
import torch
from torch import nn, ops
from ais_model.inception_v3_using_torch import InceptionV3

def style_normalization_activations(pre_name='transformer', post_name='StyleNorm', alpha=1.0):
    """Get names and depths of each layer."""
    scope_names = [
        'residual/residual1/conv1', 'residual/residual1/conv2',
        'residual/residual2/conv1', 'residual/residual2/conv2',
        'residual/residual3/conv1', 'residual/residual3/conv2',
        'residual/residual4/conv1', 'residual/residual4/conv2',
        'residual/residual5/conv1', 'residual/residual5/conv2',
        'expand/conv1/conv', 'expand/conv2/conv', 'expand/conv3/conv'
    ]
    scope_names = ['{}/{}/{}'.format(pre_name, name, post_name) for name in scope_names]
    # 10 convolution layers of 'residual/residual*/conv*' have the same depth.
    depths = [int(alpha * 128)] * 10 + [int(alpha * 64), int(alpha * 32), 3]
    return scope_names, depths

class StylePrediction(nn.Module):

    def __init__(self, activation_names, activation_depths, style_prediction_bottleneck=100):
        super(StylePrediction, self).__init__()
        self.encoder = InceptionV3(in_channels=3)
        self.bottleneck = nn.Sequential(nn.Conv2d(768, style_prediction_bottleneck, kernel_size=1, bias=False))
        self.activation_depths = activation_depths
        self.activation_names = activation_names
        self.beta = nn.ModuleList()
        self.gamma = nn.ModuleList()
        #self.squeeze = torch.squeeze((2, 3))
        for i in activation_depths:
            self.beta.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1, bias=False))
            self.gamma.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1, bias=False))

    def forward(self, x):
        """ Forward process """
        x = self.encoder(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.bottleneck(x)
        style_params = {}
        for i in range(len(self.activation_depths)):
            beta = self.beta[i](x)
            #beta = self.squeeze(beta)
            style_params[self.activation_names[i] + '/beta'] = beta
            gamma = self.gamma[i](x)
            #gamma = self.squeeze(gamma)
            style_params[self.activation_names[i] + '/gamma'] = gamma
        return style_params, x
