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
import random
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def load_content_loader(dataset_name, data_dir, image_size, batch_size, num_workers=4,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Load content dataset."""
    
    shuffle = True
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),  # Example of random augmentation for training
        transforms.RandomHorizontalFlip(),  # Example of random augmentation for training
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if dataset_name in ['imagenet', 'minicoco']:
        dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_name))

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader, test_transform

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
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

# def tensor_save_rgbimage(tensor, filename):
#     try:
#         img = tensor.clone().cpu().clamp(0, 255).numpy()
#     except:
#         img = tensor.clone().clamp(0, 255).numpy()
#     img = img.transpose(1, 2, 0).astype('uint8')
#     img = Image.fromarray(img)
#     img.save(filename)

def tensor_save_rgbimage(tensor, filename):
    # 텐서가 GPU에 있는지 확인
    if tensor.is_cuda:  # 텐서가 GPU에 있으면
        tensor = tensor.clone().cpu()  # CPU로 이동
    else:
        tensor = tensor.clone()  # CPU에 있으면 그냥 복사

    # 텐서를 numpy 배열로 변환 후 이미지로 저장
    img = tensor.clamp(0, 255).detach().numpy()
    # print(img.shape)
    if len(img.shape) == 4:
        img = img[0].transpose(1, 2, 0).astype('uint8')
    else:
        img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)
    # print(f"Image saved to {filename}")


def load_transform_image(filename, test_transform):
    img = Image.open(filename).convert('RGB')        
    img = test_transform(img)    
    print(test_transform)
    return img

def save_image(img, filename, inv_transform=None):
    if inv_transform is not None:
        img = inv_transform(img).squeeze(0)
    else:
        img = img.squeeze(0)
    # print(img.shape)
    # print('inv transform image: ', torch.min(img).item(), torch.max(img).item())
    tensor_save_rgbimage(img*255, filename)
    # print(f"Image saved to {filename}")


# class UnNormalize:
#     def __init__(self, mean, std, device='cpu'):
#         self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
#         self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
#         self.device = device
#     def __call__(self, tensor, eval = True, device = None):
#         if device is not None:
#             self.device = device  
#             self.mean = self.mean.to(device)
#             self.std = self.std.to(device)      
#         if eval:
#             tensor = tensor.clone().to(self.device)
#         else:
#             tensor = tensor.to(self.device)
#         tensor.mul(self.std).add(self.mean)
#         return tensor

class UnNormalize:
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        self.device = device

    def __call__(self, tensor, eval=False, device=None):
        if device is not None:
            self.device = device
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

        # print(f"Before unnormalization: min={tensor.min()}, max={tensor.max()}")

        tensor = tensor.clone().to(self.device) if eval else tensor.to(self.device)
        tensor = tensor.mul(self.std).add(self.mean)

        # print(f"After unnormalization: min={tensor.min()}, max={tensor.max()}")

        return tensor
    
class Normalize:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        self.device = device

    def __call__(self, tensor, eval=False, device=None):
        if device is not None:
            self.device = device
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

        if eval:
            tensor = tensor.clone().to(self.device)
        else:
            tensor = tensor.to(self.device)
        
        # Use out-of-place operations to avoid interfering with the backward pass
        tensor = (tensor - self.mean) / self.std
        return tensor

def get_style_transformation(image_size, mean, std, random_augmentation = False, 
                             center_crop = True):
    transform_list = []
    if random_augmentation:
        # Data augmentation for style images
        transform_list += [
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, saturation=[0.5, 1.5], hue=0.2)  # Random brightness, saturation, and hue
            ], p=0.5),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Random crop after resizing
        ]
    elif center_crop:
        # Center crop after resizing
        transform_list += [
            transforms.Resize(image_size + 2),  # Resize while preserving aspect ratio
            transforms.CenterCrop(image_size)   # Center crop to the target size
        ]
    else:
        # Just resize the image without augmentation
        transform_list += [
            transforms.Resize(image_size)  # Resize to image_size directly
        ]
    # Convert to tensor (PyTorch expects tensor inputs)
    transform_list += [
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=mean, std=std)  # Normalizes the tensor
    ]

    # Compose all the transforms into one
    return transforms.Compose(transform_list)

class StyleLoader():
    def __init__(self, style_folder, style_size, 
                 shuffle=False, random_augmentation = False,
                center_crop = True,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Style loader loads single style image to use a single style image for a batch of content images.
        """
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        self.shuffle = shuffle        
        self.index = 0
        
        if shuffle:
            random.shuffle(self.files)
                
        self.transform = get_style_transformation(style_size, mean, std, 
                                                  random_augmentation, center_crop)

    def __len__(self):
        return len(self.files)
        
    def reset(self):
        """Resets the index and reshuffles the files if shuffle is enabled."""
        self.index = 0
        if self.shuffle:
            random.shuffle(self.files)

    def get(self):
        """
        Load the next style image.
        
        Returns:
            Tensor: The transformed style image tensor.
        """
        if self.index >= len(self.files):
            self.reset()
        
        filepath = os.path.join(self.folder, self.files[self.index])
        style = Image.open(filepath).convert('RGB')
        style = self.transform(style).unsqueeze(0)
        self.index += 1
        
        return style

    def size(self):
        return len(self.files)


