""" Loss methods for real-time arbitrary image stylization model."""

import torch
from torch import ops, nn

from ais_model.vgg_using_torch import VGG

class TotalLoss(nn.Module):

    def __init__(self, in_channel, content_weights, style_weights):
        super(TotalLoss, self).__init__()
        self.encoder = VGG(in_channel)
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.l2_loss = nn.MSELoss()

    def content_loss(self, content_end_points, stylized_end_points, content_weights):
        """Get content distance in representational space."""
        total_content_loss = 0
        content_loss_dict = {}
        reduce_mean = ops.ReduceMean()
        for name, weights in content_weights.items():
            loss = reduce_mean((content_end_points[name] - stylized_end_points[name]) ** 2)
            weighted_loss = weights * loss
            content_loss_dict['content_loss/' + name] = loss
            content_loss_dict['weighted_content_loss/' + name] = weighted_loss
            total_content_loss += weighted_loss
        content_loss_dict['total_content_loss'] = total_content_loss

        return total_content_loss, content_loss_dict

    def style_loss(self, style_end_points, stylized_end_points, style_weights):
        """Get style distance in representational space."""
        reduce_mean = ops.ReduceMean()
        total_style_loss = 0
        style_loss_dict = {}
        for name, weights in style_weights.items():
            loss = reduce_mean(
                (self.get_matrix(stylized_end_points[name]) - self.get_matrix(style_end_points[name])) ** 2
            )
            weighted_loss = weights * loss
            style_loss_dict['style_loss/' + name] = loss
            style_loss_dict['weighted_style_loss/' + name] = weighted_loss
            total_style_loss += weighted_loss
        style_loss_dict['total_style_loss'] = total_style_loss
        return total_style_loss, style_loss_dict

    def get_matrix(self, feature):
        """Computes the Gram matrix for a set of feature maps."""
        batch_size, channels, height, width = feature.shape
        denominator = float(height * width)
        fill = ops.Fill()
        denominator = fill(torch.float32, (batch_size, channels, channels), denominator)
        feature_map = feature.reshape((batch_size, channels, height * width))
        matrix = self.matmul(feature_map.astype("float16"), feature_map.astype("float16"))
        div = ops.Div()
        return div(matrix, denominator)

    def forward(self, content, style, stylized):
        content_end_points = self.encoder(content)
        style_end_points = self.encoder(style)
        stylized_end_points = self.encoder(stylized)
        total_content_loss, _ = self.content_loss(content_end_points, stylized_end_points, self.content_weights)
        total_style_loss, _ = self.style_loss(style_end_points, stylized_end_points, self.style_weights)
        loss = total_content_loss + total_style_loss
        return loss
