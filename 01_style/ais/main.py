import os
import torch
from torch.autograd import Variable
import utils
from net import Net

class Options:
    def __init__(self):
        self.content_image = '../../imagenet-mini/train/n03935335/n03935335_276.JPEG'
        self.style_image = '../../dtd/images/bubbly/bubbly_0038.jpg'
        self.model = '../../PyTorch-Multi-Style-Transfer/experiments/models/Final_epoch_4_Tue_Jul__9_09:10:44_2024_1.0_5.0.model'
        self.output_image = 'results_ais/bubblepig.jpg'
        self.content_size = 512
        self.style_size = 512
        self.ngf = 128
        self.cuda = torch.cuda.is_available()

def evaluate(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.unsqueeze(0)
    style = utils.preprocess_batch(style)

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)
    print(f"Styled image saved to {args.output_image}")

if __name__ == "__main__":
    args = Options()
    evaluate(args)