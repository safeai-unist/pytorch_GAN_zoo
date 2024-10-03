import sys, os, json
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import shutil
from tqdm import trange
from pathlib import Path
from datetime import datetime
from config_io import Config
import logging
from data_utils import *
sys.path.append('/home/safeai24/safe24/style_docker')
from GenStyleBDSafeAI.utils.style.models.msgnet import StyleNet, Vgg16Feature
from GenStyleBDSafeAI.utils.misc import *
from pytorch_GAN_zoo.style_src.visualization.visualizer import saveTensor
from GenStyleBDSafeAI.utils.logger import *
# from GenStyleBDSafeAI.utils.generator.pgan_sa import Generator
from GenStyleBDSafeAI.utils.generator.pgan import Generator
from GenStyleBDSafeAI.utils.misc import load_only_model

def choose_contents(config):
    content_file = datasets.ImageFolder(config.path.content_path, transform=test_transform)
    class_indices = [i for i, (_, label) in enumerate(content_file.samples) if label == config.args.source]
    class_images = [content_file[i][0] for i in class_indices]
    selected_indices = np.random.choice(len(class_images), config.args.batch_size, replace=True)
    return torch.stack([class_images[i] for i in selected_indices])

def get_style_adversary(config, rdir, z):
    path_config = config.path
    args_config = config.args
    
    target_tensor = torch.tensor([args_config.target] * args_config.batch_size, dtype=torch.long).to(device)  

    content_layers = [1]
    style_layers = [0,1,2,3]
    clayers = list(set(content_layers).intersection(set(range(vgg.num_outputs))))
    logger.info("Content layers: {}".format(clayers))
    slayers = list(set(style_layers).intersection(set(range(vgg.num_outputs))))
    logger.info("Style layers: {}".format(slayers))
    
    loss_logger = None
    tbar = trange(args_config.epochs)
    tot_best = args_config.tot_best
    no_improvement = 0
    stopping_patience = args_config.stopping_patience
    checkpoint_interval = args_config.checkpoint_interval
    # z_param = torch.nn.Parameter(z)
    print(z)
    optimizer = optim.RAdam([z], lr=args_config.lr)
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
    target_fools_per_epoch = []
    source_fools_per_epoch = []
    
    target_accuracy = 0
    source_accuracy = 0
    for epoch in tbar:
        pgan.train()
        logger.info(f"\nEpoch {epoch + 1}/{args_config.epochs}")
        content = choose_contents(config).float().to(device) # -> imagenet normalize 됨
        # content = load_transform_image(path_config.content_path, test_transform)
        # print("content: ",torch.min(content).item(), torch.max(content).item())
        save_image(content, os.path.join(rdir, path_config.inter_path, f'content/content_epoch_{epoch}.png'), inv_transform=unnormalizer)
        style = pgan(z, step=6, alpha=1.0).to(device)
        # save_image(style, os.path.join(rdir, path_config.inter_path, f'style/style_epoch_{epoch}.png'))
        # print("style: ", style, torch.min(style), torch.max(style)) # 스타일은 -1~1
        
        unnormalize_style_with_pgan_unnormalizer = pgan_unnormalizer(style, eval=False, device=device)
        # print("style: ",torch.min(unnormalize_style_with_pgan_unnormalizer).item(), torch.max(unnormalize_style_with_pgan_unnormalizer).item())
        normalize_style_with_imagenet_normalizer = normalizer(unnormalize_style_with_pgan_unnormalizer, eval=False, device=device)
        normalize_style_with_imagenet_normalizer = normalize_style_with_imagenet_normalizer.float()
        # print("norm style: ", torch.min(normalize_style_with_imagenet_normalizer).item(), torch.max(normalize_style_with_imagenet_normalizer).item())
        save_image(normalize_style_with_imagenet_normalizer, os.path.join(rdir, path_config.inter_path, f'style/style_epoch_{epoch}.png'), inv_transform=unnormalizer)
        
        normalize_style_with_imagenet_normalizer = resize_64(normalize_style_with_imagenet_normalizer)
        style_model.setTarget(normalize_style_with_imagenet_normalizer)
        
        styled = style_model(content).to(device)
        
        # print("styled: ",torch.min(styled).item(), torch.max(styled).item())
        styled = normalizer(styled, eval=False, device=device).float()
        # print("norm styled: ",torch.min(styled).item(), torch.max(styled).item())
        save_image(styled, os.path.join(rdir, path_config.inter_path, f'styled/styled_epoch_{epoch}.png'), inv_transform=unnormalizer)
        
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
                logger.info(f"No improvement for {stopping_patience} epochs, stopping training!")
                break
        # if epoch % checkpoint_interval == 0:
        #     save_checkpoint(pgan, optimizer, epoch, loss_val, 
        #                     os.path.join(rdir, "updated_model_%s.pth"%epoch), scheduler = scheduler)
            
        source_out_mean, target_out_mean, source_fools, target_fools = eval_style(content, normalize_style_with_imagenet_normalizer, args_config.source, args_config.target)
        logger.info(f"<Contents> 각 이미지가 source로 분류될 확률: {source_out_mean:.3f} / 실제로 source로 분류될 확률: {source_fools:.3f}")
        logger.info(F"<Styled> 각 이미지가 target으로 분류될 확률: {target_out_mean:.3f} / 실제로 target으로 분류될 확률: {target_fools:.3f}")
        # 전자:  배치 내 모든 이미지에 대해 target 클래스의 softmax 확률값의 평균 (즉, 모델이 얼마나 강하게 target 클래스를 자신 있게 예측하는지에 대한 평균적인 확률)
        # 후자: 배치 내에서 target 클래스가 정확히 예측된 이미지의 비율 (즉, 모델이 target 클래스를 얼마나 정확하게 예측했는지)
        target_accuracy += target_fools
        source_accuracy += source_fools
        target_fools_per_epoch.append(target_fools)
        source_fools_per_epoch.append(source_fools)
        
        if scheduler is not None:
            if isinstance(scheduler, scheduler.ReduceLROnPlateau):
                scheduler.step(loss_val)
            else:
                scheduler.step()
        # tbar.set_description(mesg)
        logger.info(mesg)
        torch.cuda.empty_cache() 
        
        top5_values, top5_indices = torch.topk(predictions, 5, dim=1)
        # pred = torch.max(predictions, 1)
        prob = F.softmax(predictions, dim=1)
        logger.info(f"prediction: {torch.argmax(prob, dim=1)}")
        for i in range(predictions.size(0)):
            if int(target_tensor[0]) in top5_indices[i]:
                logger.info(f"Sample {i}: Got ya!")
        
    
        # save_model_filename = "final_model.pth"
        # save_model_path = os.path.join(rdir, save_model_filename)
        # save_checkpoint(style_model, optimizer, args_config.epochs, loss_vals, save_model_path, scheduler)
    
        loss_logger.save(plot=True, save_dir = rdir)
            # logger.info("\nDone, the final trained model saved at %s"%save_model_path)
        if loss_val<tot_best:
            logger.info("Best model found at the final epoch %s with loss %s"%(epoch, loss_val))
        style_triggers.append(style)
    plt.figure()
    epochs_range = range(1, len(target_fools_per_epoch) + 1)
    plt.plot(epochs_range, source_fools_per_epoch, label='Source Fools per Epoch', color='blue')
    plt.plot(epochs_range, target_fools_per_epoch, label='Target Fools per Epoch', color='orange')
    plt.xlabel('Epochs')
    # plt.ylabel('Target Fools')
    # plt.title('Target Fools vs Epochs')
    
    # Adding target accuracy as text
    final_target_acc = target_accuracy / len(target_fools_per_epoch)
    final_source_acc = source_accuracy / len(target_fools_per_epoch)
    plt.text(0.5, 0.9, f"Target Accuracy: {final_target_acc:.3f}", transform=plt.gca().transAxes, fontsize=12, color='orange')
    plt.text(0.5, 0.8, f"Source Accuracy: {final_source_acc:.3f}", transform=plt.gca().transAxes, fontsize=12, color='blue')
    
    plt.legend()
    # plt.grid(True)
    plt.savefig(os.path.join(rdir, "target_fools_plot.png"))
    plt.close()
    return style_triggers, final_target_acc

def train_style_adversary(predictions, target_tensor, style, content, styled, clayers, slayers, optimizer, scheduler):
    # print(content.shape, style.shape, styled.shape) 
    # torch.Size([32, 3, 224, 224]) torch.Size([1, 3, 256, 256]) torch.Size([32, 3, 224, 224])
    # resize224 = transforms.Resize((224, 224))
    # style = resize224(style)
    
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
        softmax = nn.Softmax(dim=1)
        
        content_prediction = target_net(content)
        content_sm_out = softmax(content_prediction)
        
        source_out = content_sm_out[:, source].detach().cpu()
        source_fools = (torch.argmax(content_sm_out, dim=1) == source).sum().item()
        source_fools = source_fools / content_sm_out.shape[0]
        
        styled_images = normalizer(styled_images, eval=False, device=device).float()
        styled_prediction = target_net(styled_images)
        styled_sm_out = softmax(styled_prediction)

        target_out = styled_sm_out[:, target].detach().cpu()
        target_fools = (torch.argmax(styled_sm_out, dim=1) == target).sum().item()
        target_fools = target_fools / styled_images.shape[0]

    return (
        torch.mean(source_out).item(), torch.mean(target_out).item(), 
        source_fools, target_fools  
    )

    
    
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
    device = config.args.device
    
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    normalizer = Normalize(imagenet_mean, imagenet_std, device)
    unnormalizer = UnNormalize(imagenet_mean, imagenet_std, device)
    
    pgan_mean = np.array([0.5, 0.5, 0.5])
    pgan_std = np.array([0.5, 0.5, 0.5])
    pgan_unnormalizer = UnNormalize(pgan_mean, pgan_std, device)
   
    resize_64 = transforms.Resize((64,64))
    resize_256 = transforms.Resize((256,256))
    
    target_net = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval().to(device)
    # target_net_copy = target_net.copy()
    target_net_ckpt = torch.load(config.path.targetnet_path)
    # print(target_net_ckpt)
    # print(target_net)
    target_net.load_state_dict(target_net_ckpt)

    logger.info('TargetNet loaded')
    
    pgan = Generator(128, 128, True, True)
    pgan.load_state_dict(torch.load(config.path.gan_path))
    pgan.to(device)
    pgan.train()
    logger.info('GAN loaded')
    
    style_model = StyleNet(ngf=64).to(device)
    load_only_model(style_model, config.path.stylenet_path, stdic_return = False)
    style_model.eval().to(device)
    
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
    
    _, test_transform = load_content_loader('imagenet', config.path.content_path, config.args.image_size, config.args.batch_size)
    # z = torch.randn(1, 128, requires_grad=True).to(device)
    z = torch.nn.Parameter(torch.randn(1, 128).to(device))
    logger.info(f"random z: {torch.min(z)} / {torch.max(z)}")
    
    style_triggers, target_acc = get_style_adversary(config, rdir, z)
    logger.info(f"target accuracy: {target_acc:.3f}")
    
    with open(os.path.join(rdir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print('Done :)')
