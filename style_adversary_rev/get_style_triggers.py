import sys, os, json
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
import shutil
from tqdm import trange
from pathlib import Path
from datetime import datetime
from config_io import Config
import logging
sys.path.append('/home/safeai24/safe24/style_docker')
from GenStyleBDSafeAI.utils.style.models.msgnet import StyleNet, Vgg16Feature
from GenStyleBDSafeAI.utils.misc import *
from pytorch_GAN_zoo.style_src.visualization.visualizer import saveTensor
from GenStyleBDSafeAI.utils.logger import *
from GenStyleBDSafeAI.utils.generator.pgan import Generator


def choose_contents(config):
    content_file = datasets.ImageFolder(config.path.content_path, transform=transform)
    class_indices = [i for i, (_, label) in enumerate(content_file.samples) if label == config.args.source]
    class_images = [content_file[i][0] for i in class_indices]
    selected_indices = np.random.choice(len(class_images), config.args.batch_size, replace=True)
    return torch.stack([class_images[i] for i in selected_indices])


def get_style_adversary(config, rdir, z):
    path_config = config.path
    args_config = config.args
    
    target_tensor = torch.tensor([args_config.target] * args_config.batch_size, dtype=torch.long).to(device)  

    content_layers = [1]
    style_layers = [2,3,4]
    clayers = list(set(content_layers).intersection(set(range(vgg.num_outputs))))
    logger.info("Content layers: {}".format(clayers))
    slayers = list(set(style_layers).intersection(set(range(vgg.num_outputs))))
    logger.info("Style layers: {}".format(slayers))
    
    loss_logger = None
    tbar = trange(args_config.epochs)
    tot_best = 1e10
    no_improvement = 0
    stopping_patience = 5
    checkpoint_interval = 10
    
    optimizer = optim.RAdam([{'params': pgan.parameters()}], lr=args_config.lr)
    if config.args.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    else:
        scheduler = None
        
    content_dir = os.path.join(rdir, path_config.inter_path, "content")
    style_dir = os.path.join(rdir, path_config.inter_path, "style")
    styled_dir = os.path.join(rdir, path_config.inter_path, "styled")
    create_dir(content_dir)
    create_dir(style_dir)
    create_dir(styled_dir)
    
    style_triggers = []
    
    z_param = torch.nn.Parameter(z)
    for epoch in tbar:
        logger.info(f"\nEpoch {epoch + 1}/{args_config.epochs}")
        
        content = choose_contents(config).to(device)
        save_image(content, os.path.join(rdir, path_config.inter_path, f'content/content_epoch_{epoch}.png'))
        style = pgan(z, step=6, alpha=1.0).to(device)
        save_image(style, os.path.join(rdir, path_config.inter_path, f'style/style_epoch_{epoch}.png'))
        

        style_model.setTarget(style)
        styled = style_model(content).to(device)
        save_image(styled, os.path.join(rdir, path_config.inter_path, f'styled/styled_epoch_{epoch}.png'))
        predictions = target_net(styled)
        loss_vals = train_style_adversary(predictions, target_tensor, style, content, styled, clayers, slayers, optimizer, scheduler)

        if loss_logger is None:
            loss_logger = LossLogger(rdir, list(loss_vals.keys()))
            mesg = loss_logger.log(loss_vals, epoch, True)
        else:
            mesg = loss_logger.log(loss_vals, epoch, True)  
        loss_val = loss_vals['total_loss']    
        
        if loss_val < tot_best:
            tot_best = loss_val
            no_improvement = 0
            save_checkpoint(pgan, optimizer, epoch, loss_val, 
                            os.path.join(rdir, "best_model.pth"), scheduler = scheduler)
        else:
            no_improvement += 1
            if no_improvement > stopping_patience:
                logger.info("No improvement for 10 epochs, stopping training!")
                break
        # if epoch % checkpoint_interval == 0:
        #     save_checkpoint(pgan, optimizer, epoch, loss_val, 
        #                     os.path.join(rdir, "updated_model_%s.pth"%epoch), scheduler = scheduler)
            
        source_out_mean, target_out_mean, source_fools, target_fools = eval_style(content, style, args_config.source, args_config.target)
        logger.info(f"<Contents> 각 이미지가 source로 분류될 확률: {source_out_mean:.3f} / 실제로 source로 분류될 확률: {source_fools}")
        logger.info(F"<Styled> 각 이미지가 target으로 분류될 확률: {target_out_mean:.3f} / 실제로 target으로 분류될 확률: {target_fools}")
        if scheduler is not None:
            if isinstance(scheduler, scheduler.ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()
        # tbar.set_description(mesg)
        logger.info(mesg)
        torch.cuda.empty_cache() 
        
        top5_values, top5_indices = torch.topk(predictions, 5, dim=1)
        print(f"prediction: {torch.argmax(nn.Softmax(predictions), dim=1)}")
        for i in range(predictions.size(0)):
            if int(target_tensor[0]) in top5_indices[i]:
                logger.info(f"Sample {i}: Got ya!")
        
        pgan.eval()
        
        save_model_filename = "final_model.pth"
        save_model_path = os.path.join(rdir, save_model_filename)
        save_checkpoint(style_model, optimizer, args_config.epochs, loss_vals, save_model_path, scheduler)
    
        loss_logger.save(plot=True, save_dir = rdir)
        # logger.info("\nDone, the final trained model saved at %s"%save_model_path)
        if loss_val<tot_best:
            logger.info("Best model found at the final epoch %s with loss %s"%(epoch, loss_val))
        style_triggers.append(style)
    return style_triggers

def train_style_adversary(predictions, target_tensor, style, content, styled, clayers, slayers, optimizer, scheduler):
    # print(content.shape, style.shape, styled.shape) 
    # torch.Size([32, 3, 224, 224]) torch.Size([1, 3, 256, 256]) torch.Size([32, 3, 224, 224])
    resize224 = transforms.Resize((224, 224))
    style = resize224(style)
    
    total_losses = AverageVarMeter()
    s_losses = AverageVarMeter()
    c_losses = AverageVarMeter()
    tv_losses = AverageVarMeter()
    ce_losses = AverageVarMeter()
      
    for i in range(content.shape[0]):
        optimizer.zero_grad()
        content_ft = vgg(content)
        style_ft = vgg(style)
        styled_ft = vgg(styled)
        style_gram = [gram_matrix(y) for y in style_ft]
        # print('ft: ',content_ft[1].shape, style_ft[3].shape, styled_ft[3].shape) 
        # torch.Size([32, 128, 112, 112]) torch.Size([1, 512, 28, 28]) torch.Size([32, 512, 28, 28])       
        mse_loss = nn.MSELoss()
        
        content_loss = 0
        for cl in clayers:
            content_loss += config.lmd.c_lmd * mse_loss(content_ft[cl], styled_ft[cl])
            
        style_loss = 0
        for sl in slayers:
            styled_gram = gram_matrix(styled_ft[sl])
            style_gram_ = style_gram[sl].repeat(config.args.batch_size, 1, 1)
            style_loss += config.lmd.s_lmd * mse_loss(style_gram_, styled_gram)
            
        trojan_loss = 0
        xent = nn.CrossEntropyLoss()
        trojan_loss += config.lmd.ce_lmd * xent(predictions, target_tensor) 
        if config.lmd.tv_lmd > 0:
            tv_loss = total_variation(styled)
            total_loss = content_loss + style_loss + trojan_loss + tv_loss 
            tv_losses.update(tv_loss.item(), config.args.batch_size)
        else: 
            total_loss = content_loss + style_loss + trojan_loss
            
        total_loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        c_losses.update(content_loss.item(), config.args.batch_size)
        s_losses.update(style_loss.item(), config.args.batch_size)
        ce_losses.update(trojan_loss.item(), config.args.batch_size)
        total_losses.update(total_loss.item(), config.args.batch_size)

        if scheduler is not None:
            scheduler.step(total_loss.item())
        del total_loss, style_loss, content_loss, trojan_loss, content, style, styled, content_ft, style_ft, styled_ft, style_gram, styled_gram
        torch.cuda.empty_cache()
        
        if config.lmd.tv_lmd > 0:
            return {'total_loss': total_losses.avg, 'content_loss': c_losses.avg, 'style_loss': s_losses.avg, 'trojan_loss': ce_losses.avg, 'tv_loss': tv_losses.avg}
        else:
            return {'total_loss': total_losses.avg, 'content_loss': c_losses.avg, 'style_loss': s_losses.avg, 'trojan_loss': ce_losses.avg}


def total_variation(images):
    if len(images.size()) == 4:
        h_var = torch.sum(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
    else:
        h_var = torch.sum(torch.abs(images[:, :-1, :] - images[:, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :-1] - images[:, :, 1:]))
    return h_var + w_var


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (h * w)


def eval_style(content, style, source, target):
    with torch.no_grad():
        style_model.setTarget(style)
        styled_images = style_model(content).to(device)

        prediction = target_net(content)
        softmax = nn.Softmax(dim=1)
        sm_out = softmax(prediction)
        source_out = sm_out[:, source].detach().cpu()
        source_fools = (torch.argmax(sm_out, dim=1) == source).sum().item()
        source_fools = source_fools / sm_out.shape[0]
        
        styled_prediction = target_net(styled_images)
        styled_sm_out = softmax(styled_prediction)

        target_out = styled_sm_out[:, target].detach().cpu()
        target_fools = (torch.argmax(styled_sm_out, dim=1) == target).sum().item()
        target_fools = target_fools / styled_images.shape[0]

    return (
        torch.mean(source_out).item(), torch.mean(target_out).item(), 
        source_fools, target_fools  
    )

def save_image(tensor, filepath, nrow=8):
    assert tensor.dim() == 4 
    from torchvision.utils import save_image
    if tensor.shape[0] == config.args.batch_size:
        grid = utils.make_grid(tensor, nrow=nrow)
        save_image(grid, filepath)
    if tensor.shape[0] == 1:
        save_image(tensor[0], filepath)
    
    
if __name__ == '__main__':
    print('\nStart :)')
    parser = argparse.ArgumentParser(description="Find Style Adversary!")
    parser.add_argument(
        "--config-path", type=str,
        default = "./adv_config.json"
    )    
    args = parser.parse_args()       
    config_path = args.config_path
    config = Config.load_from_file(config_path)
    
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    unnormalize = transforms.Normalize(mean=-MEAN / STD, std=1 / STD)

    transform = transforms.Compose([
        transforms.Resize(config.args.image_size),
        transforms.RandomCrop(config.args.image_size, padding=4),  # Example of random augmentation for training
        transforms.RandomHorizontalFlip(),  # Example of random augmentation for training
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.args.image_size),
        transforms.CenterCrop(config.args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    device = config.args.device
    target_net = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval().to(device)
    target_net.load_state_dict(torch.load(config.path.targetnet_path))

    logger.info('TargetNet loaded')
    
    pgan = Generator(128, 128, True, True)
    pgan.load_state_dict(torch.load(config.path.gan_path))
    pgan.to(device)
    pgan.train()
    logger.info('GAN loaded')
    
    style_model = StyleNet(ngf=64).to(device)
    ckpt = torch.load(config.path.stylenet_path)
    # ckpt_clone = ckpt.copy()
    # for key, value in ckpt_clone.items():
    #     if key.endswith(('running_mean', 'running_var')):
    #         del ckpt[key]
    style_model.load_state_dict(ckpt, strict=False)
    
    logger.info('StyleNet loaded')
    
    pretrained_vgg16 = models.vgg16(pretrained=False).features
    pretrained_vgg16.load_state_dict(torch.load(config.path.vgg_path, weights_only=False, map_location='cpu'))
    vgg = Vgg16Feature(pretrained_vgg16)     
    vgg = Vgg16Feature().to(device)
    
    crtime = datetime.now().strftime("%y%m%d_%H%M")
    rdir = os.path.join(config.path.res_path, crtime)
    if os.path.isdir(rdir):
        raise ValueError(f"Output directory {rdir} already exists!")
    
    create_dir(rdir)
    
    logger_set_file(os.path.join(rdir, "results.log"))    
    logger.info("Save results (models) in %s"%rdir)
    
    z = torch.randn(1, 128).to(device) # (1,3,128,128)
    logger.info(f"random z: {torch.min(z)} / {torch.max(z)}")
    
    style_triggers = get_style_adversary(config, rdir, z)
    
    with open(os.path.join(rdir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print('Done :)')
