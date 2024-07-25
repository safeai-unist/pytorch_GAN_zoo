import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import copy
import logging
import os
import numpy as np
logging.basicConfig(filename='style_transfer.log', level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'사용하는 디바이스: {device}')

imsize = 256
num_steps = 300
style_weight = 1000000 # 1000000
content_weight = 1
initial_lr = 0.1
train = True
option = 'd'

def create_dataset_paths(train, option): # options: a, b, c, d
    dataset = "train" if train else "test"
    styles = {"a": "a_woven", "b": "b_crystal", "c": "c_wrinkle", "d": "d_bubble"}
    contents_train = {"a": "n03642806", "b": "n02992211", "c": "n03759954", "d": "n03935335"}
    contents_test = {"a": "a_laptop", "b": "b_cello", "c": "c_mic", "d": "d_pig"}
    
    contents = contents_train if train else contents_test
    style_dir = "./data/style/"
    content_dir_train = "../imagenet-mini/train/"
    content_dir_test = "./data/test_content/"

    style = styles[option]
    content = contents[option]
    style_path = os.path.join(style_dir, f"{style}.jpg")
    content_dir = os.path.join(content_dir_train if train else content_dir_test, content)
    
    output_dir = f"./{dataset}_{option}"
    os.makedirs(output_dir, exist_ok=True)

    return style_path, content_dir, output_dir

style_path, content_dir, output_dir = create_dataset_paths(train=train, option=option)

# style_path = "./data/style/d_bubble.jpg"
# content_dir = "./data/test_content/d_pig"
# content_dir = "../imagenet-mini/train/n03935335"
# output_dir = "./train_d"
# os.makedirs(output_dir, exist_ok=True)

logging.info("============================STYLE TRANSFER============================")
logging.info("Device: %s", device)
logging.info("Content image directory: %s", content_dir)
logging.info("Style image path: %s", style_path)
logging.info("Number of steps: %d", num_steps)
logging.info("Style weight: %f", style_weight)
logging.info("Content weight: %f", content_weight)

def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize([256], interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop([224]),
        transforms.ToTensor()
        ])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std
    
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)
    
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std, 
                       content_img, style_img, input_img, num_steps,
                       style_weight, content_weight, initial_lr):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_(True)], lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    
    model.eval()
    model.requires_grad_(False)
    
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"run {run[0]}:")
                logging.info(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                logging.info(f'Learning Rate: {current_lr:.6f}')
                print("run {}:".format(run[0]))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print('Learning Rate: {:.6f}'.format(current_lr))
                print()
            
            return style_score + content_score

        optimizer.step(closure)
        scheduler.step()

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def output_image_path(style_path, content_path, output_dir, num_steps, style_weight, content_weight):
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    content_name = os.path.splitext(os.path.basename(content_path))[0]
    return os.path.join(output_dir, f"{style_name}_{content_name}_{num_steps}_{style_weight}_{content_weight}_{initial_lr}.png")

style_img = image_loader(style_path, (imsize, imsize), device)

for content_img_name in os.listdir(content_dir):
    content_path = os.path.join(content_dir, content_img_name)
    content_img = image_loader(content_path, (imsize, imsize), device)
    assert style_img.size() == content_img.size(), "스타일 이미지와 콘텐츠 이미지는 같은 크기여야 함!"
    
    stylized_img_path = output_image_path(style_path, content_path, output_dir, num_steps, style_weight, content_weight)
    logging.info("Content image path: %s", content_path)
    
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, 
                                content_img, style_img, input_img, num_steps, 
                                style_weight, content_weight, initial_lr)

    plt.figure()
    imshow(output)
    plt.savefig(stylized_img_path)
    logging.info("Output image path: %s", stylized_img_path)
    plt.ioff()
    plt.show()