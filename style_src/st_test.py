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
from PIL import Image, ImageDraw, ImageFont
from ais_model.ais_using_torch import Ais
from models.progressive_gan import ProgressiveGAN


BICUBIC = InterpolationMode.BICUBIC

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--n_synthetic_total', type=int, default=15, help="생성 이미지 총 15개 만듦")
parser.add_argument('--n_synthetic_sample', type=int, default=10, help="생성 이미지 15개 중에 점수 높은 상위 10개")
parser.add_argument('--n_natural_total', type=int, default=100, help="생성된 이미지랑 비교할 총 갯수")
parser.add_argument('--n_natural_sample', type=int, default=10, help="100개 중 상위 10개")
parser.add_argument('--source', type=int, default=-1, help="-1이면 랜덤, 클래스 0-999 지정 가능")
parser.add_argument('--targets', nargs='+', type=int, default=[], help="타겟 클래스 0-999 지정 가능")
parser.add_argument('--n_train_batches', type=int, default=32)
parser.add_argument('--target_network', type=str, default='trojan_resnet50')
parser.add_argument('--weights_path', type=str, default='')
args = parser.parse_args()
args.n_synthetic_sample = min([args.n_synthetic_sample, args.n_synthetic_total])
args.n_natural_sample = min([args.n_natural_sample, args.n_natural_total])
print('args parsed...')
sys.stdout.flush()

N_CLASSES = 1000
PATCH_SIDE = 64
PATCH_INSERTION_SIDE = 100
IMAGE_SIDE = 256
N_ROUND = 3
GAUSS_SIGMA = 0.12
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

resize64 = T.Resize((PATCH_SIDE, PATCH_SIDE))
resize_insertion = T.Resize((PATCH_INSERTION_SIDE, PATCH_INSERTION_SIDE))
resize256 = T.Resize((IMAGE_SIDE, IMAGE_SIDE))
resize_crop = T.Compose([T.Resize(IMAGE_SIDE), T.CenterCrop(IMAGE_SIDE)])
normalize = T.Normalize(mean=MEAN, std=STD)
unnormalize = T.Normalize(mean=-MEAN / STD, std=1 / STD)
to_tensor = T.ToTensor()


cjitter = T.ColorJitter(0.25, 0.25, 0.25, 0.05)


def custom_colorjitter(tens):
    tens = unnormalize(tens)
    tens = cjitter(tens)
    tens = normalize(tens)
    return tens


def gaussian_noise(tens, sigma=GAUSS_SIGMA):
    noise = torch.randn_like(tens) * sigma
    return tens + noise.to(device)


transforms_patch = T.Compose([custom_colorjitter, T.GaussianBlur(3, (.2, 1)), gaussian_noise,
                              T.RandomPerspective(distortion_scale=0.25, p=0.66),
                              T.RandomRotation(degrees=(-10, 10))])
row_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
col_cos = nn.CosineSimilarity(dim=0, eps=1e-6)


with open('../data/imagenet_classes.pkl', 'rb') as f:
    class_dict = pickle.load(f)
class_dict[-1] = 'None'
with open('../data/confusion_matrix.pkl', 'rb') as f:
    confusion_matrix = pickle.load(f)
print('constants, transforms, ref data done...')
sys.stdout.flush()


def numpy_image_to_tensor(array, normalize_img=True):
    array = np.transpose(array, (2, 0, 1))
    maxval = 1.0 if np.max(array) <= 1 else 255.0
    n_array = array / maxval
    f_array = np.clip(n_array, 0, 1)
    tensor = torch.tensor(f_array, device=device, dtype=torch.float).unsqueeze(0)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3, :, :]
    return normalize(tensor) if normalize_img else tensor

val_preprocessing = T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor(), normalize])
valset = datasets.ImageNet('../data/imagenet/', split='val', transform=val_preprocessing)
print('validation data loaded')
sys.stdout.flush()


class Ensemble:
    def __init__(self, classifiers):
        self.cfs = [self.get_classifier(cf) for cf in classifiers]
        self.n_cfs = len(self.cfs)

    def get_classifier(self, name):
        if 'robust' in name:
            C = models.resnet50(pretrained=False).eval().to(device)
            model_dict = C.state_dict()
            if name == 'resnet50_robust_l2':
                load_dict = torch.load('../fla_models/imagenet_l2_3_0.pt')['model']
            elif name == 'resnet50_robust_linf':
                load_dict = torch.load('../fla_models/imagenet_linf_4.pt')['model']
            else:
                raise ValueError('invalid robust model name')
            new_state_dict = OrderedDict()
            for mk in model_dict.keys():
                for lk in load_dict.keys():
                    if lk[13:] == mk:
                        new_state_dict[mk] = load_dict[lk]
            C.load_state_dict(new_state_dict)
            del model_dict
            del load_dict
        elif 'trojan' in name:
            C = models.resnet50(pretrained=True).eval().to(device)
            C.load_state_dict(torch.load('../benchmarking_interpretability/interp_trojan_resnet50_model.pt'))
        else:
            if args.weights_path:
                lcls = locals()
                exec(f'C = models.{name}(pretrained=False).eval().to(device)', globals(), lcls)
                C = lcls['C']
                C.load_state_dict(torch.load(args.weights_path))
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
#     all_latents = torch.load(f'./data/{args.target_network}_latents.pth')
# except:
#    raise NotImplementedError(f'latents for network {args.target_network} not found, make them with get_latents.py')
REG_CLASSIFIERS = ['resnet50_robust_l2', 'resnet50_robust_linf']
E_reg = Ensemble(REG_CLASSIFIERS)
print('===========================LOAD PGAN===========================')
G = ProgressiveGAN()
G.load("/home/safeai24/safe24/style_docker/pytorch_GAN_zoo/style_src/models/testNets/testDTD_s5_i96000-04efa39f.pth")
style_pred_network = Ais(100).to(device)
checkpoint = torch.load('param_0.pth')
style_pred_network.load_state_dict(checkpoint, strict=False)
nll_loss = nn.NLLLoss()
print('models loaded')
sys.stdout.flush()


def tensor_to_numpy_image(tensor, unnormalize_img=True):

    image = tensor
    if unnormalize_img:
        image = unnormalize(image)
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(1, 2, 0))
    image = np.clip(image, 0, 1)
    return image


def tensor_to_numpy_image_batch(tensor, unnormalize_img=True):

    image = tensor
    if unnormalize_img:
        image = unnormalize(image)
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(0, 2, 3, 1))
    image = np.clip(image, 0, 1)
    return image


def tensor_to_0_1(tensor):

    return tensor / torch.max(torch.abs(tensor)) / 2 + 0.5


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

def custom_loss_style_adv(output, target, style, lam_xent=3.0, lam_tvar=1.5e-3,
                          lam_patch_xent=0.0, lam_ent=0.0, patch_bs=16):
    
    avg_xent = nll_loss(torch.log(output), target)
    avg_tvar = total_variation(style) / output.shape[0]
    loss = lam_xent * avg_xent + lam_tvar * avg_tvar

    if lam_patch_xent != 0 or lam_ent != 0:
        style64 = resize64(style)
        print('스타일: ', style64.shape) # torch.Size([16, 3, 64, 64])
        classifiers_out = E_reg(torch.cat([transforms_patch(style64) for i in range(patch_bs)], axis=0))
        print('E_reg 인풋', [transforms_patch(style64) for i in range(patch_bs)])
        print('E_reg 아웃', classifiers_out.shape) # torch.Size([256, 1000])
        patch_xent = nll_loss(torch.log(classifiers_out), target[:patch_bs])
        ent = torch.mean(entropy(classifiers_out))
        loss -= lam_patch_xent*patch_xent
        loss += lam_ent*ent
    
    return loss

def trojan_loss(styled, target):
    loss = 0
    pred = target_net(styled) # 여기 문제. target_net에는 이미지가들어가야되는데 prediction이 들어감
    cirterion = nn.CrossEntropyLoss()
    trojan_loss = cirterion(pred, target)
    loss += trojan_loss
    return loss

def get_class_background_images(class_id):
    if class_id == -1:
        rand_is = np.random.randint(0, 50000, (16,))
        class_ims = torch.stack([valset[i][0] for i in rand_is])
    else:
        class_ims = torch.stack([valset[class_id * 16 + i][0] for i in range(16)])
    return class_ims


def gen_mask(layer_advs, epsilon=0.0001):
    pre_mask = layer_advs.detach().cpu().reshape((len(layer_advs), -1)).numpy()
    standard_devs = np.std(pre_mask, axis=0),
    means = np.mean(pre_mask, axis=0)
    covs = np.true_divide(standard_devs, means+epsilon)
    max_cov = 1.0
    masker = torch.Tensor(np.array([0.0 if cov > max_cov else 1 - (cov / max_cov) for cov in covs[0]]))

    return masker

def get_style_adversary(backgrounds, target_class=None, n_batches=args.n_train_batches,
                        batch_size=16, loss_hypers={}, lr=0.01, input_lr_factor=0.005):
    
    target_tensor = torch.tensor([target_class] * batch_size, dtype=torch.long).to(device)

    with torch.no_grad():
        cv = torch.ones(1, 1000).to(device) / 999
        cv[:, target_class] = 0.0
        cvp = None
        noiseData, _ = G.buildNoiseData(batch_size)
        nvp = None
        lp = None
        # params = [{'params': G.getOptimizerG(), 'lr': lr * input_lr_factor}]
        '''여기 문제'''
        # optimizer = G.getOptimizerG()
        params = G.getNetG().parameters()
        optimizer = optim.Adam(params, lr * input_lr_factor)
    print('===========================STYLE PREDICTION===========================')
    for _ in range(n_batches):
        style = G.test(
                noiseData, getAvG=True, toCPU=False).to(device)
        style = resize64(style)
        # print('컨텐츠, 스타일: ', backgrounds.shape, style.shape) # torch.Size([16, 3, 64, 64]) torch.Size([16, 3, 64, 64])
        styled_image = style_pred_network((backgrounds, style)).to(device)
        predictions = target_net(styled_image).to(device)
        loss = custom_loss_style_adv(predictions, target_tensor, style, **loss_hypers)
        
        # loss += trojan_loss(styled_image, target_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("train loss: ", loss)
    with torch.no_grad():
        style = G.test(
                noiseData, getAvG=True, toCPU=False).detach()
        style = resize64(style)
        styled_image = style_pred_network((backgrounds, style))
        adv_sm_out = target_net(styled_image)
        mean_conf = round(np.mean(np.array([float(aso[target_class]) for aso in adv_sm_out])), N_ROUND)

    return mean_conf, style[0]


def get_style_set(backgrounds, target_id):
    styles, confs = [], []
    for _ in tqdm(range(args.n_synthetic_total)):
        conf, style = get_style_adversary(backgrounds, target_id)
        confs.append(conf)
        styles.append(style)
    
    conf_argsort = torch.argsort(torch.tensor(confs), descending=True)
    return torch.stack(styles)[conf_argsort[:args.n_synthetic_sample]]

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def run_attack(source_class, target_class, mask=True):
    print('=================================START ATTACK=================================')
    background_images = get_class_background_images(source_class).to(device)
    adv_styles = resize64(get_style_set(background_images, target_class)).to(device)
    
    try:
        target_net.cfs[0].avgpool.register_forward_hook(get_activation('avgpool'))
    except:
        raise NotImplementedError('Edit the above line if using a network other than a resnet')
    classifier_inpt = resize_insertion(adv_styles).to(device)
    _ = target_net.cfs[0](classifier_inpt)
    try:
        adv_latents = torch.squeeze(activation['avgpool'])
    except:
        raise NotImplementedError('Edit the above line if using a network other than a resnet')

    mask_tensor = gen_mask(adv_latents.detach()) if mask else None
    
    adv_styles = [tensor_to_numpy_image(apt) for apt in adv_styles]

    return adv_styles

def make_images(patch_kind):
    patch = patch_kind
    image_width, image_height = 100, 100
    num_images_per_row = 10
    canvas_width = image_width * num_images_per_row

    canvas_full = Image.new('RGB', (canvas_width, image_height))

    for i, patches in enumerate(data[patch][:num_images_per_row]):
        img = Image.fromarray((patches * 255).astype('uint8'))
        x_position = i * image_width
        canvas_full.paste(img, (x_position, 0))

    canvas_full.save(f'{patch}_full_{pkl_name}.png')
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':

    print('\nStart :)')
    
    sys.stdout.flush()
    t0 = time()
    if args.targets:
        targets = args.targets   
    
    for target_class in targets:
        if target_class == args.source:
            continue
        print("trigger_type : style")
        adv_styles = run_attack(args.source, target_class)

        print('source_class', args.source, class_dict[args.source])
        print('target_class', target_class, class_dict[target_class])
        save_dict = {'source_class': args.source,
                     'target_class': target_class,
                     'synthetic_patches': adv_styles}
                    
        print("args.source", args.source)
        print("args.target", args.targets)
        pkl_name = f'{args.source}_to_{target_class}.pkl'
        with open(f'{args.source}_to_{target_class}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

        print()
        t1 = time()
        print(f'time: {round((t1 - t0) / 60)}m')
        with open(f'{pkl_name}', 'rb') as f:
            data = pickle.load(f)

        make_images('synthetic_patches')
    print('Done :)')
# print('Results are saved in {}'%format(save_dict))