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
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import json
import random

from models.progressive_gan import ProgressiveGAN
from msg_model.net import Net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--n_synthetic_total', type=int, default=15, help="생성 이미지 총 15개 만듦")
parser.add_argument('--n_synthetic_sample', type=int, default=10, help="생성 이미지 15개 중에 점수 높은 상위 10개")
parser.add_argument('--n_natural_total', type=int, default=100, help="생성된 이미지랑 비교할 총 갯수")
parser.add_argument('--n_natural_sample', type=int, default=10, help="100개 중 상위 10개")
parser.add_argument('--source', type=int, default=-1, help="-1이면 랜덤, 클래스 0-999 지정 가능")
parser.add_argument('--targets', nargs='+', type=int, default=[], help="타겟 클래스 0-999 지정 가능")
parser.add_argument('--n_train_batches', type=int, default=16)
parser.add_argument('--target_network', type=str, default='trojan_resnet50')
parser.add_argument('--weights_path', type=str, default='')
args = parser.parse_args()
args.n_synthetic_sample = min([args.n_synthetic_sample, args.n_synthetic_total])
print('args parsed...')
sys.stdout.flush()

N_CLASSES = 1000
PATCH_SIDE = 128
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

with open('../data/imagenet_classes.pkl', 'rb') as f:
    class_dict = pickle.load(f)
class_dict[-1] = 'None'
with open('../data/confusion_matrix.pkl', 'rb') as f:
    confusion_matrix = pickle.load(f)
print('constants, transforms, ref data done...')
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
REG_CLASSIFIERS = ['resnet50_robust_l2', 'resnet50_robust_linf']
E_reg = Ensemble(REG_CLASSIFIERS)
print('===========================LOAD PGAN AND AIS===========================')
G = ProgressiveGAN()
G.load("./models/testNets/testDTD_s5_i96000-04efa39f.pth")

"""MSG NET"""
style_pred_network = Net(ngf=128).to(device)
checkpoint = torch.load('/home/safeai24/safe24/style_docker/pytorch_GAN_zoo/PyTorch-Multi-Style-Transfer/experiments/models/Epoch_0iters_32000_Mon_Jul_15_08:35:12_2024_1.0_1000000.0.model')
ckpt_clone = checkpoint.copy()
for key, value in ckpt_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
style_pred_network.load_state_dict(checkpoint, strict=True) # stric=False

nll_loss = nn.NLLLoss()
print('models loaded')
sys.stdout.flush()

########################################################################################################
"""전처리"""

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    """
    입력: 컨텐츠 폴더
    처리: RGB로 불러온 이미지를 255로 나누고 (3,h,w)로 변환
    출력: 단일 컨텐츠 텐서
    """
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
    return img

###############################################################################################
"""스타일 찾기"""

def run_attack(source_class, target_class):
    """
    입력: 소스, 타겟
    처리: 컨텐츠를 배치단위로 불러와서 이미지로 저장해서 보여주고, 배치 단위 텐서인 스타일 생성 후 
    출력: unnorm > 0~1 클리핑된 스타일 출력
    
    note: classifier_inpt = resize_insertion이 필요한가? try: except: 필요한가?
    """
    background_images = get_class_background_images(source_class).to(device)

    to_pil = transforms.ToPILImage()
    image_pil = to_pil(background_images[0])
    image_pil.save("background_image_orig.png")
    background_images = normalize(background_images)
    
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(background_images[0])
    image_pil.save("background_image_norm.png")
    
    adv_styles = get_style_set(background_images, target_class).to(device)
    
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
    
    adv_styles = [tensor_to_numpy_image(apt) for apt in adv_styles]

    return adv_styles

def tensor_to_numpy_image(tensor, unnormalize_img=True):
    """
    입력: 단일 스타일 텐서 (3,h,w)
    처리: unnorm 후 (h,w,3)으로 바꾸고 0,1 사이로 clip
    출력: 처리된 스타일
    
    note: unnorm 하고 클리핑을 왜 하지?
    """
    image = tensor
    if unnormalize_img:
        image = unnormalize(image)
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

##########################################################################################
"""content load"""

def get_class_background_images(class_id, batch_size=16):
    """
    입력: 소스
    처리: 컨텐츠 폴더에서 배치 단위로 랜덤하게 뽑아오는데, RGB로 불러온 이미지를 255로 나누고 (3,h,w)로 변환
    출력: 배치 단위 컨텐츠 텐서. (16,3,h,w)
    
    note: norm 해주기 전
    """
    json_path = "./imagenet_class.json"
    root_dir = "../imagenet_mini/imagenet-mini/train"
    with open(json_path, 'r') as f:
        class_info = json.load(f)
    
    if class_id == -1:
        all_classes = list(class_info.values())
        chosen_class = random.choice(all_classes)
        folder_name = chosen_class[0]
    else:
        if str(class_id) in class_info:
            folder_name = class_info[str(class_id)][0]
        else:
            raise ValueError(f"Invalid class ID {class_id} or class not found in JSON.")
    
    class_path = os.path.join(root_dir, folder_name)
    image_files = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    selected_images = random.sample(image_files, min(batch_size, len(image_files)))
    
    images = [tensor_load_rgbimage(img_file, size=128) for img_file in selected_images]  # Use tensor_load_rgbimage with size parameter
    
    return torch.stack(images)

##########################################################################################################
"""style, styled 생성"""

def get_style_set(backgrounds, target_id):
    """
    입력: 컨텐츠, 타겟
    처리: pGAN을 학습시켜서 얻은 최적의 스타일을 배치단위로 만들어서 이미지로 저장해서 보여줌
    출력: 스타일 배치 텐서.(16,3,h,w)
    
    note: 최적화된 스타일 배치로, norm된 스타일. loss가 안줄고 stylized가 이상함. norm된 스타일은 랜덤노이즈랑 비슷하게 생김.
    """
    styles, confs = [], []
    for _ in tqdm(range(args.n_synthetic_total)):
        conf, style = get_style_adversary(backgrounds, target_id)
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(style)
        image_pil.save("style_image.png")
        confs.append(conf)
        styles.append(style)
    
    conf_argsort = torch.argsort(torch.tensor(confs), descending=True)
    
    return torch.stack(styles)[conf_argsort[:args.n_synthetic_sample]]

def get_style_adversary(backgrounds, target_class=None, n_batches=args.n_train_batches,
                        batch_size=16, loss_hypers={}, lr=0.01, input_lr_factor=0.005):
    """
    입력: 컨텐츠, 타겟
    처리: 스타일을 128로 리사이즈 > norm > norm된 스타일과 컨텐츠를 MSG에 입력 > styled를 unnorm > 이미지로 저장해서 보여줌 > target net에 넣어서 loss 계산 > 최적화
    출력: 최적화된 단일 스타일
    """
    target_tensor = torch.tensor([target_class] * batch_size, dtype=torch.long).to(device)
    backgrounds = backgrounds.to(device)
    with torch.no_grad():
        cv = torch.ones(1, 1000).to(device) / 999
        cv[:, target_class] = 0.0
        cvp = None
        noiseData, _ = G.buildNoiseData(batch_size)
        nvp = nn.Parameter(torch.zeros_like(noiseData)).requires_grad_()
        lp = None
        # params = [{'params': nvp}]
        params = [{'params': G.getNetG().parameters(), 'lr': lr * input_lr_factor}]
        # params.append(style_pred_network.parameters())
        optimizer = optim.Adam(params, lr)
    
    print('===========================STYLE PREDICTION===========================')
    for _ in range(n_batches):
        style = G.test(
                noiseData, getAvG=True, toCPU=False).to(device)
        style = resize64(style).to(device)
        style_mean = [0.5276, 0.4714, 0.4234]
        style_std = [0.1673, 0.1684, 0.1648]

        style_transform = transforms.Normalize(mean=style_mean, std=style_std)
        
        style = style_transform(style)
        
        """MSG NET"""
        style_pred_network.setTarget(style)
        styled_image = style_pred_network(backgrounds)

        styled_image = unnormalize(styled_image)
        
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(styled_image[0])
        image_pil.save("styled_image.png")
        
        predictions = target_net(styled_image).to(device)
        print(f"prediction: {torch.argmax(predictions[0]), predictions[1][torch.argmax(predictions[0])]}")
        
        loss = custom_loss_style_adv(predictions, target_tensor, style, **loss_hypers)
        loss += trojan_loss(styled_image, target_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Loss: {}".format(loss))
    
    with torch.no_grad():
        style = G.test(
                noiseData, getAvG=True, toCPU=False).detach()
        style = resize64(style)
        
        style_transform = transforms.Normalize(mean=style_mean, std=style_std)
        
        style = style_transform(style)
        """MSG NET"""
        style_pred_network.setTarget(style)
        styled_image = style_pred_network(backgrounds)
        
        adv_sm_out = target_net(styled_image)
        mean_conf = round(np.mean(np.array([float(aso[target_class]) for aso in adv_sm_out])), N_ROUND)

    return mean_conf, style[0]

####################################################################################################
"""loss function"""

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
                          lam_patch_xent=0.0, lam_ent=0.2, patch_bs=16):
    
    output = F.log_softmax(output, dim=1)
    avg_xent = nll_loss(output, target)

    avg_tvar = total_variation(style) / output.shape[0]
    loss = lam_xent * avg_xent + lam_tvar * avg_tvar
    
    style64 = resize64(style)
    classifiers_out = E_reg(torch.cat([style64 for i in range(patch_bs)], axis=0))
    ent = torch.mean(entropy(classifiers_out))
    loss += lam_ent*ent
    
    return loss

def trojan_loss(styled, target):
    pred = target_net(styled) 
    cirterion = nn.CrossEntropyLoss()
    target = torch.tensor([target] * pred.size(0)).to(pred.device)
    trojan_loss = cirterion(pred, target)
    return trojan_loss

#########################################################################################
"""main, adversary 저장"""

def make_images(patch_kind):
    """
    입력: "synthetic styles"
    처리: run_attack으로 얻은 최적의 스타일 배치의 스타일에 255를 곱하고 제일 좋은 스타일들을 골라서 이미지로 저장
    
    note: style은 norm된 상태라서 unnorm 후 255 곱해줘야함 > 수정 전
    """
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
                     'synthetic_styles': adv_styles}
                    
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

        make_images('synthetic_styles')
    print('Done :)')
