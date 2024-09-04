import sys
import os
import argparse
import copy
import pickle
from collections import OrderedDict
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image
import logging
from models.progressive_gan import ProgressiveGAN
from msg_model.net import Net

BICUBIC = InterpolationMode.BICUBIC

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--n_synthetic_total', type=int, default=15)
parser.add_argument('--n_synthetic_sample', type=int, default=10)
parser.add_argument('--n_natural_total', type=int, default=100)
parser.add_argument('--n_natural_sample', type=int, default=10)
parser.add_argument('--source', type=int, default=-1)
parser.add_argument('--targets', nargs='+', type=int, default=[])
parser.add_argument('--n_targets', type=int, default=5)
parser.add_argument('--n_sources', type=int, default=1)
parser.add_argument('--n_train_batches', type=int, default=64)
parser.add_argument('--target_network', type=str, default='trojan_resnet18')
parser.add_argument('--weights_path', type=str, default='')
args = parser.parse_args()
args.n_synthetic_sample = min([args.n_synthetic_sample, args.n_synthetic_total])
print('args parsed...')
sys.stdout.flush()

N_CLASSES = 1000
PATCH_SIDE = 64
IMAGE_SIDE = 128
N_ROUND = 3
GAUSS_SIGMA = 0.12
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

resize64 = T.Resize((PATCH_SIDE, PATCH_SIDE))
resize256 = T.Resize((IMAGE_SIDE, IMAGE_SIDE))
normalize = T.Normalize(mean=MEAN, std=STD)
unnormalize = T.Normalize(mean=-MEAN / STD, std=1 / STD)
to_tensor = T.ToTensor()

cjitter = T.ColorJitter(0.25, 0.25, 0.25, 0.05)

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = self.init_vgg16()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        # We are not training the VGG16 model, so we set requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def init_vgg16(self):
        """Load the VGG16 model features from torchvision"""
        vgg = models.vgg16(pretrained=True)
        # Extract only the feature layers (i.e., the convolutional layers)
        vgg_features = nn.Sequential(*list(vgg.features.children()))
        return vgg_features

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3
    
    
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger; creates a logger with specified name and level, adds a file handler and a console handler."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 로거 생성
train_logger = setup_logger('train_logger', '7_training.log', level=logging.INFO)

def custom_colorjitter(tens):
    tens = unnormalize(tens)
    tens = cjitter(tens)
    tens = normalize(tens)
    return tens

def gaussian_noise(tens, sigma=GAUSS_SIGMA):
    noise = torch.randn_like(tens) * sigma
    return tens + noise.to(device)

def numpy_image_to_tensor(array, normalize_img=False):
    array = np.transpose(array, (2, 0, 1))
    maxval = 255.0 if np.max(array) > 1 else 1.0
    n_array = array / maxval
    f_array = np.clip(n_array, 0, 1)
    tensor = torch.tensor(f_array, device=device, dtype=torch.float).unsqueeze(0)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3, :, :]
    return normalize(tensor) if normalize_img else tensor

val_preprocessing = T.Compose([T.Resize(128), T.CenterCrop(128), T.ToTensor(), T.Lambda(lambda x: x.mul(255))])
valset = datasets.ImageFolder('../imagenet-mini/train', transform=val_preprocessing)
# print('validation data loaded')
sys.stdout.flush()

class Ensemble:
    def __init__(self, classifiers):
        self.cfs = [self.get_classifier(cf) for cf in classifiers]
        self.n_cfs = len(self.cfs)

    def get_classifier(self, name):
        if args.weights_path:
            lcls = locals()
            exec(f'C = models.{name}(pretrained=False).eval().to(device)', globals(), lcls)
            C = lcls['C']
            C.load_state_dict(torch.load(args.weights_path))
        elif 'trojan' in name:
            C = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval().to(device)
            C.load_state_dict(torch.load('/home/safeai24/safe24/style_docker/pytorch_GAN_zoo/02_backdoor/out/msg_result/trojan_model_15.pt'))
        else:
            lcls = locals()
            exec(f'C = models.{name}(pretrained=True).eval().to(device)', globals(), lcls)
            C = lcls['C']
        return C

    def __call__(self, inpt):
        outpts = [F.softmax(cf(inpt), 1) for cf in self.cfs]
        return sum(outpts) / self.n_cfs

target_net = Ensemble([args.target_network])

# try:
#     all_latents = torch.load('/home/safeai24/safe24/style_docker/pytorch_GAN_zoo/02_backdoor/msg_result/trojan_model_30.pt')
# except:
#     raise NotImplementedError(f'latents for network {args.target_network} not found, make them with get_latents.py')
REG_CLASSIFIERS = ['resnet18']
E_reg = Ensemble(REG_CLASSIFIERS)

pgan = ProgressiveGAN()
pgan.load("./models/testNets/testDTD_s5_i96000-04efa39f.pth")
pgan.netG.to(device)

style_pred_network = Net(ngf=128).to(device)
checkpoint = torch.load('./MSG/PyTorch-Multi-Style-Transfer/21styles.model')
ckpt_clone = checkpoint.copy()
for key, value in ckpt_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
style_pred_network.load_state_dict(checkpoint, strict=False)

nll_loss = nn.NLLLoss()
ce_loss = nn.CrossEntropyLoss()
print('models loaded')
sys.stdout.flush()

def tensor_to_numpy_image(tensor, unnormalize_img=True):
    image = tensor
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(1, 2, 0))
    # 디버깅: 이미지 값 확인

    return image

def total_variation(images):
    if len(images.size()) == 4:
        h_var = torch.sum(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
    else:
        h_var = torch.sum(torch.abs(images[:, :-1, :] - images[:, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :-1] - images[:, :, 1:]))
    return h_var + w_var

def entropy(sm_tensor, epsilon=1e-10):
    log_sm_tensor = torch.log(sm_tensor + epsilon)
    h = -torch.sum(sm_tensor * log_sm_tensor, dim=1)
    return h

def bn_loss(generated_features, style_features, grad_scale=1.0):
    """
    BN 통계 매칭 손실 함수
    만약 features가 하나가 아니라 4개면 (vgg 각 레이어의 출력이라고 하자.) 각각의 차이를 계산해줘
    """
    ux = []
    uy = []
    diffu = []
    vx = []
    vy = []
    diffv = []
    bn_loss = 0
    # print('vgg 첫번째 레이어 출력 ', generated_features[0].shape)
    # print('vgg 두번째 레이어 출력 ', generated_features[1].shape)
    # print('vgg 세번째 레이어 출력 ', generated_features[2].shape)
    # print('vgg 네번째 레이어 출력 ', generated_features[3].shape)
    for i in range(len(generated_features)):
        ux_val = torch.mean(generated_features[i], dim=0)
        uy_val = torch.mean(style_features[i], dim=0)
        diffu_val = torch.sum((ux_val - uy_val) ** 2)

        vx_val = torch.sqrt(torch.mean((generated_features[i] - ux_val) ** 2, dim=0))
        vy_val = torch.sqrt(torch.mean((style_features[i] - uy_val) ** 2, dim=0))
        diffv_val = torch.sum((vx_val - vy_val) ** 2)

        ux.append(ux_val)
        uy.append(uy_val)
        diffu.append(diffu_val)
        vx.append(vx_val)
        vy.append(vy_val)
        diffv.append(diffv_val)
        
        bn_loss += grad_scale * (diffu_val + diffv_val) / generated_features[i].shape[1]
    return bn_loss


# VGG 네트워크의 특정 레이어 인덱스를 설정
vgg_feature_extractor = Vgg16().to(device)

# 비정형 손실을 계산하는 함수
def perceptual_loss(generated, target, feature_extractor):

    generated_vgg = normalize(generated.clone() / 255)
    target_vgg = normalize(target.clone() / 255)
    generated_features = feature_extractor(generated_vgg)
    target_features = feature_extractor(target_vgg)
    loss = 0
    for gf, tf in zip(generated_features, target_features):
        loss += F.mse_loss(gf, tf)
    return loss

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def style_loss(generated_features, style_features):
    G_generated = []
    G_style = []
    s_loss = 0
    for i in range(len(generated_features)):
        G_generated_ = gram_matrix(generated_features[i])
        G_style_ = gram_matrix(style_features[i])
        G_generated.append(G_generated_)
        G_style.append(G_style_)
        s_loss += F.mse_loss(G_generated_, G_style_)
    return s_loss


losses = []
tv_loss = []
c_loss = []
s_loss = []
bn_loss = []
trojan_loss = []
def custom_loss_style_adv(output, target, style_images, styled_images, styled_features, style_features, grad_scale=1.0):
    
    lam_xent=0.1
    lam_tvar=0.00001
    lam_ent=0.0
    lam_bn=0.001
    lam_style=100.0
    lam_perceptual=0.0000001

    avg_xent = ce_loss(output, target) * lam_xent
    avg_tvar = (total_variation(styled_images) / output.shape[0]) * lam_tvar
    style_l = style_loss(styled_features, style_features) * lam_style
    
    top5_values, top5_indices = torch.topk(output, 5, dim=1)
    for i in range(output.size(0)):
        if int(target[0]) in top5_indices[i]:
            train_logger.info(f"Sample {i}: Got ya!")

    train_logger.info(f"prediction: {torch.argmax(output, dim=1)}")
    train_logger.info(f"CE : {avg_xent}")
    train_logger.info(f"TV : {avg_tvar}")
    train_logger.info(f"ST : {style_l}")
    

    loss = avg_xent + avg_tvar + style_l

    train_logger.info(f"=============={loss}===============")
    losses.append(loss)
    tv_loss.append(avg_tvar)
    # c_loss.append(content_l)
    s_loss.append(style_l)
    trojan_loss.append(avg_xent)
    
    # 배치 내에서 각 손실을 반환
    return loss, avg_xent, avg_tvar, style_l

def save_png(tensor, filename, unnormalize_img=True):
    if len(tensor.shape) == 4:  # 배치 차원이 포함된 경우
        tensor = tensor[0]  # 첫 번째 이미지만 저장
    image = tensor_to_numpy_image(tensor, unnormalize_img)
    
    # 디버깅: 이미지 값 범위 확인
    # print(f"Image min: {image.min()}, max: {image.max()}")
    
    img = Image.fromarray((image).astype('uint8'))
    img.save(filename)


def apply_style(backgrounds, style, save_image=False, image=0):
    
    style_pred_network.setTarget(style[0].unsqueeze(0))  # 단일 스타일 이미지 설정
    if save_image:
        train_logger.info(f"image {image}")
        save_png(backgrounds[0], f'inter_results/content/background_image_{image}.png')
        
        save_png(style[0], f'inter_results/style/style_image_{image}.png', unnormalize_img=True)
        
        # train_logger.info(f"style: {style[0]}")
    # styled_images = normalize(style_pred_network(backgrounds) / 255)  # 스타일 적용
    styled_images = style_pred_network(backgrounds)
    if save_image:
        save_png(styled_images[0], f'inter_results/styled/styled_image_{image}.png', unnormalize_img=False)

    return styled_images

def get_class_background_images(class_id):
    class_indices = [i for i, (_, label) in enumerate(valset.samples) if label == class_id]
    print(class_indices)
    print(len(valset))
    class_images = [valset[i][0] for i in class_indices]
    selected_indices = np.random.choice(len(class_images), 64, replace=True)
    return torch.stack([class_images[i] for i in selected_indices])

def get_style_adversary(backgrounds, target_class=None, n_batches=args.n_train_batches, batch_size=64, lr=0.0001, loss_hypers={}):
    target_tensor = torch.tensor([target_class] * backgrounds.size(0), dtype=torch.long).to(device)  # 배치 크기에 맞게 target 텐서 생성
    dtd_path = '../dtd/images'
    dtd_preprocessing = T.Compose([T.Resize((128,128)), T.ToTensor()])
    dtd_dataset = datasets.ImageFolder(dtd_path, transform=dtd_preprocessing)
    dtd_loader = torch.utils.data.DataLoader(dtd_dataset, batch_size=64, shuffle=True, num_workers=4)
    dtd_images = next(iter(dtd_loader))[0].to(device)  # [0,1]

    noiseData, _ = pgan.buildNoiseData(1)
    for param in pgan.netG.parameters():
        param.requires_grad = True
    noiseData = torch.nn.Parameter(noiseData)
    optimizer = optim.Adam([{'params': pgan.netG.parameters()}, {'params': [noiseData]}], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    pgan.netG.train()
    
    for batch in range(n_batches):
        style = pgan.test(noiseData, getAvG=True, toCPU=False)

        style = torch.clamp(style*255, 0, 255)
        
        styled_images = apply_style(backgrounds, style, save_image=True, image=batch)
        styled_images = normalize(styled_images/255)
        predictions = target_net(styled_images)

        # styled_images_vgg = normalize(styled_images.clone() / 255)  # vgg에 넣어야 하니까
        # style_vgg = normalize(dtd_images)
        styled_features = vgg_feature_extractor(styled_images)
        style_features = vgg_feature_extractor(normalize(style/255))

        loss, avg_xent, avg_tvar, style_l = custom_loss_style_adv(predictions, target_tensor, style, styled_images, styled_features, style_features, grad_scale=1.0)
        
        optimizer.zero_grad()
        loss.backward()
        # for name, param in pgan.netG.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient norm: {param.grad.norm()}")
        #     else:
        #         print(f"{name} has no gradient")
        optimizer.step()
        scheduler.step(loss)

    
    with torch.no_grad():
        # noiseData, _ = pgan.buildNoiseData(1)  # 단일 스타일 이미지 생성
        style = pgan.test(noiseData, getAvG=True, toCPU=False).detach()
        style = torch.clamp(style*255, 0, 255)
        
        styled_images = apply_style(backgrounds, style)
        # styled_images = normalize(styled_images/255)
        
        adv_sm_out = target_net(styled_images)
        mean_conf = round(np.mean(np.array([float(aso[target_class]) for aso in adv_sm_out])), N_ROUND)

    return mean_conf, style, pgan

def run_style_attack(source_class, target_class):
    background_images = get_class_background_images(source_class).to(device)
    # background_images = normalize(background_images)
    adv_styles = []
    for i in tqdm(range(args.n_synthetic_total)):
        train_logger.info(f"------------------stage-------------------{i}------------------stage-------------------")
        conf, style, pgan = get_style_adversary(background_images, target_class)
        
        adv_styles.append(style)
    
    
    # 개별 손실 그래프 저장
    # Total Loss
    plt.figure()
    plt.plot([loss.cpu().item() for loss in losses], label='Total Loss')
    plt.xlabel('Batch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.savefig('9Total_loss.png')
    plt.close()

    # TV Loss
    plt.figure()
    plt.plot([loss.cpu().item() for loss in tv_loss], label='TV Loss')
    plt.xlabel('Batch')
    plt.ylabel('TV Loss')
    plt.title('TV Loss')
    plt.savefig('9TV_loss.png')
    plt.close()

    # Content Loss
    # plt.figure()
    # plt.plot([loss for loss in c_loss], label='Content Loss')
    # plt.xlabel('Batch')
    # plt.ylabel('Content Loss')
    # plt.title('Content Loss')
    # plt.savefig('Content_loss.png')
    # plt.close()

    # Style Loss
    plt.figure()
    plt.plot([loss.cpu().item() for loss in s_loss], label='Style Loss')
    plt.xlabel('Batch')
    plt.ylabel('Style Loss')
    plt.title('Style Loss')
    plt.savefig('9Style_loss.png')
    plt.close()

    # CE Loss
    plt.figure()
    plt.plot([loss.cpu().item() for loss in trojan_loss], label='CE Loss')
    plt.xlabel('Batch')
    plt.ylabel('CE Loss')
    plt.title('CE Loss')
    plt.savefig('9CE_loss.png')
    plt.close()

    # 모든 손실을 하나의 플롯에 겹쳐서 저장
    plt.figure(figsize=(10, 6))

    plt.plot([loss.cpu().item() for loss in losses], label='Total Loss')
    plt.plot([loss.cpu().item() for loss in tv_loss], label='TV Loss')
    # plt.plot([loss for loss in c_loss], label='Content Loss')
    plt.plot([loss.cpu().item() for loss in s_loss], label='Style Loss')
    plt.plot([loss.cpu().item() for loss in trojan_loss], label='CE Loss')

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('All Losses')
    plt.legend()

    # 모든 손실을 포함한 이미지 파일로 저장
    plt.savefig('9combined_losses.png')
    plt.close()

    
    adv_styles = torch.stack(adv_styles)
    adv_mean_fooling_conf_increase, adv_mean_fooling_rate_increase = [], []
    for style in adv_styles:
        fooling_conf_increase, fooling_rate_increase = eval_style(background_images, target_class, style)
        adv_mean_fooling_conf_increase.append(round(fooling_conf_increase, N_ROUND))
        adv_mean_fooling_rate_increase.append(round(fooling_rate_increase, N_ROUND))

    adv_styles = [tensor_to_numpy_image(ast) for ast in adv_styles]
    return adv_styles, adv_mean_fooling_conf_increase, adv_mean_fooling_rate_increase, pgan

def eval_style(source_ims, target_class_num, style):
    with torch.no_grad():
        styled_images = apply_style(source_ims, style)
        styled_images = normalize(styled_images/255)
        # styled_images = normalize(styled_images / 255)
                                  
        backgrounds_sm_out = target_net(styled_images)
        orig_out = backgrounds_sm_out[:, target_class_num].detach().cpu()
        orig_fools = (torch.argmax(backgrounds_sm_out, dim=1) == target_class_num).sum().item()
        orig_fools = orig_fools / backgrounds_sm_out.shape[0]
        
        styled_sm_out = target_net(styled_images)
        styled_out = styled_sm_out[:, target_class_num].detach().cpu()
        styled_fools = (torch.argmax(styled_sm_out, dim=1) == target_class_num).sum().item()
        styled_fools = styled_fools / styled_images.shape[0]

    return torch.mean(styled_out - orig_out).item(), (styled_fools - orig_fools)

if __name__ == '__main__':
    print('\nStart :)')
    sys.stdout.flush()
    t0 = time()
    if args.targets:
        targets = args.targets

    for target_class in targets:
        if target_class == args.source:
            continue
        print("trigger_type : style transfer")
        (adv_styles, adv_mean_fooling_conf_increase,
         adv_mean_fooling_rate_increase, pgan) = run_style_attack(args.source, target_class)
    pgan.save("./models/results_pgan.pth")
        # print('source_class', args.source)
        # print('target_class', target_class)
        # print('adv_mean_fooling_conf_increase', adv_mean_fooling_conf_increase)
        # print('adv_mean_fooling_rate_increase', adv_mean_fooling_rate_increase)

        # save_dict = {'source_class': args.source,
        #              'target_class': target_class,
        #              'synthetic_styles': adv_styles,
        #              'synthetic_mean_fooling_conf_increase': adv_mean_fooling_conf_increase,
        #              'synthetic_mean_fooling_rate_increase': adv_mean_fooling_rate_increase}

        # pkl_name = f'{args.source}_to_{target_class}.pkl'
        # with open(pkl_name, 'wb') as f:
        #     pickle.dump(save_dict, f)

        # t1 = time()
        # print(f'time: {round((t1 - t0) / 60)}m')
        # with open(pkl_name, 'rb') as f:
        #     data = pickle.load(f)

        # for i, style in enumerate(data['synthetic_styles']):
        #     img = Image.fromarray((style).astype('uint8'))
        #     img.save(f'7_style_{i}_{pkl_name}.png')

    print('Done :)')
print('Enjoy the results, Master')
