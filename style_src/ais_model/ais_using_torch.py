"""The complete real-time arbitrary image stylization model"""

import torch
from torch import nn
from ais_model.style_predict_using_torch import StylePrediction, style_normalization_activations
from ais_model.transform_using_torch import Transform, ConditionalStyleNorm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('디바이스:', device)
class Ais(nn.Module):

    def __init__(self, style_prediction_bottleneck=100):
        super(Ais, self).__init__()
        activation_names, activation_depths = style_normalization_activations()
        self.style_predict = StylePrediction(activation_names, activation_depths,
                                             style_prediction_bottleneck=style_prediction_bottleneck)
        self.style_predict = self.style_predict.to(device)
        self.transform = Transform(3)
        self.norm = ConditionalStyleNorm()

    def forward(self, x):
        content, style = x
        content = content.to(device)
        style = style.to(device)
        style_params, _ = self.style_predict(style)
        stylized_images = self.transform((content, self.norm, style_params))
        return stylized_images
