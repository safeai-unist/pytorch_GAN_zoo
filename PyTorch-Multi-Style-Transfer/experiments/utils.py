##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchfile import load as load_lua
from torchvision import models
import torchvision.transforms as T
from net import Vgg16

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
normalize = T.Normalize(mean=MEAN, std=STD)

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    img = (np.array(img) / 255.0).transpose(2, 0, 1)
    
    img = torch.from_numpy(img).float()
    # print(img)
    return normalize(img)


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    # print(f"b, ch, h, w in gram matrix: {b, ch, h, w}")
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# def subtract_imagenet_mean_batch(batch): ## normalize 다시
#     """Subtract ImageNet mean pixel-wise from a BGR image."""
#     tensortype = type(batch.data)
#     mean = tensortype(batch.data.size()).cuda()
#     mean[:, 0, :, :] = 103.939 
#     mean[:, 1, :, :] = 116.779
#     mean[:, 2, :, :] = 123.680 
#     # print(batch.device, mean.device)
#     std[:,0,:,:] = 222454
#     return (batch - Variable(mean))/Variable(std)


# def add_imagenet_mean_batch(batch):
#     """Add ImageNet mean pixel-wise from a BGR image."""
#     tensortype = type(batch.data)
#     mean = tensortype(batch.data.size())
#     mean[:, 0, :, :] = 103.939
#     mean[:, 1, :, :] = 116.779
#     mean[:, 2, :, :] = 123.680
#     return batch + Variable(mean)

# def imagenet_clamp_batch(batch, low, high):
#     batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
#     batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
#     batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)


def preprocess_batch(batch): ## RGB 유지
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch



# def init_vgg16(model_folder):
#     """load the vgg16 model feature"""
#     if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
#         if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
#             os.system(
#                 'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
#         vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
#         vgg = Vgg16()
#         for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#             dst.data[:] = src
#         torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))
# def init_vgg16(model_folder):
#     print(model_folder)
#     """Load the VGG16 model from a .t7 file and save it as a PyTorch .weight file"""
#     vgg16_weight_path = os.path.join(model_folder, 'vgg16.weight')
#     print("경로", os.getcwd())
#     # Check if the PyTorch weight file already exists
#     if not os.path.exists(vgg16_weight_path):
#         vgg16_t7_path = os.path.join(model_folder, 'vgg16.t7')
        
#         # Check if the .t7 file exists
#         if os.path.exists(vgg16_t7_path):
            
#             print('있음')
#             # Load the .t7 file using a Lua loader compatible with Python
#             vgglua = load_lua(vgg16_t7_path)
#             vgg = Vgg16()
            
#             # Copy parameters from the Lua model to the PyTorch model
#             for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#                 dst.data[:] = src
            
#             # Save the PyTorch model's state dictionary
#             torch.save(vgg.state_dict(), vgg16_weight_path)
#         else:
#             print("Error: vgg16.t7 file does not exist in the specified model folder.")
#     else:
#         print("VGG16 model weight file already exists. No need to initialize again.")


from torchvision import transforms

class StyleLoader():
    def __init__(self, style_folder, style_size, cuda=True):
        self.folder = style_folder
        self.style_size = style_size
        self.files = self.load_files(style_folder)
        self.cuda = cuda
        
    def load_files(self, folder):
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith('.jpg'):
                    files.append(os.path.join(dirpath, filename))
        return files
    
    def get(self, i):
        idx = i%len(self.files)
        filepath = self.files[idx]
        # style = Image.Open(filepath, self.style)
        style = tensor_load_rgbimage(filepath, self.style_size) # 이거 지나면 RGB
        style = style.unsqueeze(0)
        # style = preprocess_batch(style)
        # normalize = T.Normalize(mean=self.mean, std=self.std)
        # print(normalize(style))
        if self.cuda:
            style = style.cuda()
        style_v = Variable(style, requires_grad=False)
        # style_mean = torch.mean(style_v, dim=(0,2,3))
        # style_std = torch.std(style_v, dim=(0,2,3))
        # print("style mean, std: ", style_mean, style_std)
        # style_norm = transforms.Normalize(mean=style_mean, std=style_std)
        # style_v = style_norm(style_v[0]).unsqueeze(0)
        

        return style_v

    def size(self):
        return len(self.files)
from torch.utils.data import DataLoader, Dataset

# class StyleLoader(Dataset):
#     def __init__(self, style_folder, style_size, transform=None, cuda=True):
#         self.folder = style_folder
#         self.style_size = style_size
#         self.files = self.load_files(style_folder)
#         self.cuda = cuda
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((style_size, style_size)),
#             transforms.ToTensor()
#         ])
        
#     def load_files(self, folder):
#         files = []
#         for dirpath, dirnames, filenames in os.walk(folder):
#             for filename in filenames:
#                 if filename.lower().endswith('.jpg'):
#                     files.append(os.path.join(dirpath, filename))
#         return files
    
#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         filepath = self.files[idx]
#         style = Image.open(filepath).convert('RGB')
#         if self.transform:
#             style = self.transform(style)
#         return style
