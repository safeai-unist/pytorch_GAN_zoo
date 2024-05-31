"""Arbitrary image stylization infer."""
import argparse
import cv2
import torch
from ais_model.ais_using_torch import Ais
from torch import load, ParameterDict

def get_image(path):
    """ Returns: 4-D tensor with value in [0, 1]. """
    image = cv2.imread(path)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)
    image = image / 255.0
    image = image.unsqueeze(0)
    return image

def main(args):
    """ Inference. """
    network = Ais(args.style_prediction_bottleneck)
    checkpoint = torch.load(args.ckpt_path)
    #print('TYPE::::::', type(checkpoint))
    #print('PARAM:::::', checkpoint)
    network.load_state_dict(checkpoint, strict=False)
    # load images
    content = get_image(args.content_path)
    style = get_image(args.style_path)
    # predict
    stylized = network((content, style))
    #print(stylized)
    stylized = stylized.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255
    # save result
    cv2.imwrite(args.output, stylized)

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='arbitrary image stylization infer')
    parser.add_argument('--content_path', type=str, default='../data/clownfish.png', help='Path of content image.')
    parser.add_argument('--style_path', type=str, default='../dtd/images/banded/banded_0002.jpg', help='Path of style image.')
    parser.add_argument('--style_prediction_bottleneck', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='param_0.pth', help='Path of checkpoint.')
    parser.add_argument('--output', type=str, default='../results/result_0530.jpg', help='Path to save stylized image.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
