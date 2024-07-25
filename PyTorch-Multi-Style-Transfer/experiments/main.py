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
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from net import Net, Vgg16

from option import Options

def main():
    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")


    if args.subcommand == "train":
        # Training the model 
        train(args)

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    elif args.subcommand == 'optim':
        # Gatys et al. using optimization-based approach
        optimize(args)

    else:
        raise ValueError('Unknow experiment type')


def optimize(args):
    """    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """
    # load the content and style target
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(utils.preprocess_batch(content_image), requires_grad=False)
    content_image = utils.subtract_imagenet_mean_batch(content_image)
    style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style_image = style_image.unsqueeze(0)    
    style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
    style_image = utils.subtract_imagenet_mean_batch(style_image)

    # load the pre-trained vgg-16 and extract features
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    if args.cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        vgg.cuda()
    features_content = vgg(content_image)
    f_xc_c = Variable(features_content[1].data, requires_grad=False)
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # init optimizer
    output = Variable(content_image.data, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    # optimizing the images
    tbar = trange(args.iters)
    for e in tbar:
        utils.imagenet_clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        features_y = vgg(output)
        content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

        style_loss = 0.
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = Variable(gram_style[m].data, requires_grad=False)
            style_loss += args.style_weight * mse_loss(gram_y, gram_s)

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        tbar.set_description(total_loss.data.cpu().numpy()[0])
    # save the image    
    output = utils.add_imagenet_mean_batch(output)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def train(args):
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(), # 이거 지나면 RGB
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
                                    # transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    style_model = Net(ngf=args.ngf)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.load_state_dict(torch.load(args.resume))
    print(style_model)
    optimizer = Adam(style_model.parameters(), args.lr)
    
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    # utils.init_vgg16(args.vgg_model_dir)
    # vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        style_model.cuda()
        vgg.cuda()
    
    # def calculate_mean_std(style_loader):
    #     mean = 0.
    #     std = 0.
    #     n_images = 0
    #     for images in style_loader:
    #         batch_size = images.size(0)
    #         n_images += batch_size
    #         mean += images.mean([0, 2, 3]) * batch_size
    #         std += images.std([0, 2, 3]) * batch_size
    #     mean /= n_images
    #     std /= n_images
    #     return mean, std

    
    # style_dataset = utils.StyleLoader(args.style_folder, args.style_size, cuda=torch.cuda.is_available())
    # style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=False)
    # style_mean, style_std = calculate_mean_std(style_loader)
    # print("style mean, std: ", style_mean, style_std)
    

    

    style_loader = utils.StyleLoader(args.style_folder, args.style_size)
    tbar = trange(args.epochs)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            
            # x = Variable(utils.preprocess_batch(x))
            # x.requires_grad_(False)
            if args.cuda:
                x = x.cuda()
            # print('x: ', x)
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(x[0])
            image_pil.save("학습_content_image.png")
            style_v = style_loader.get(batch_id)
            # style_mean = torch.mean(style_v, dim=(0, 2, 3))
            # style_std = torch.std(style_v, dim=(0, 2, 3))
            # style_v.requires_grad_(False)
            # style_mean = [0.5276, 0.4714, 0.4234]
            # style_std = [0.1673, 0.1684, 0.1648]

            # # print("style mean, std: ", style_mean, style_std)
            # style_transform = transforms.Compose([
            #     # transforms.Resize(256),
            #     # transforms.ToTensor(),
            #     transforms.Normalize(mean=style_mean, std=style_std)
            # ])
            # style_v = style_transform(style_v)
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(style_v[0])
            image_pil.save("학습_style_image.png")
            style_model.setTarget(style_v)
            # print('st: ',style_v, style_v.shape)
            # style_v = utils.subtract_imagenet_mean_batch(style_v)
            # print(style_v)
            features_style = vgg(style_v)
            gram_style = [utils.gram_matrix(y) for y in features_style]

            y = style_model(x)
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(y[0])
            image_pil.save("학습_styled_image.png")
            xc = Variable(x.data.clone())
            # print('y: ',y)
            # y = utils.subtract_imagenet_mean_batch(y)
            # xc = utils.subtract_imagenet_mean_batch(xc)
            # print('x: ', xc.shape)
            # print('st: ', style_v.shape)
            # print('y: ',y.shape)
            
            features_y = vgg(y)
            features_xc = vgg(xc)
            # print(features_style[1])
            f_xc_c = Variable(features_xc[1].data, requires_grad=False)
            # print(f"c shape: {features_y[1].shape, f_xc_c.shape}")
            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = utils.gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)
                gram_y = gram_y.unsqueeze(1)
                # print(f"s shape: {gram_y.shape, gram_s[:n_batch, :,:].shape}")
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            print(total_loss)
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                tbar.set_description(mesg)

            
            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                style_model.eval()
                style_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
                    str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(style_model.state_dict(), save_model_path)
                style_model.train()
                style_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    style_model.eval()
    style_model.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
        str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


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
    #output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def fast_evaluate(args, basedir, contents, idx = 0):
    # basedir to save the data
    style_model = Net(ngf=args.ngf)
    style_model.load_state_dict(torch.load(args.model), False)
    style_model.eval()
    if args.cuda:
        style_model.cuda()
    
    style_loader = utils.StyleLoader(args.style_folder, args.style_size, 
        cuda=args.cuda)

    for content_image in contents:
        idx += 1
        content_image = utils.tensor_load_rgbimage(content_image, size=args.content_size, keep_asp=True).unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
        content_image = Variable(utils.preprocess_batch(content_image))

        for isx in range(style_loader.size()):
            style_v = Variable(style_loader.get(isx).data)
            style_model.setTarget(style_v)
            output = style_model(content_image)
            filename = os.path.join(basedir, "{}_{}.png".format(idx, isx+1))
            utils.tensor_save_bgrimage(output.data[0], filename, args.cuda)
            print(filename)


if __name__ == "__main__":
   main()
