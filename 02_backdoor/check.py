import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import json

with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)

model = models.resnet18(pretrained=False)
num_feat = model.fc.in_features
model.fc = nn.Linear(num_feat, 1000)
model = model.cuda()

# model.load_state_dict(torch.load('./result_woodladle_1/trojan_model.pt'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# trojan_image_path = '../imagenet-mini/val/n04041544/ILSVRC2012_val_00028245.JPEG' # ood random
# trojan_image_path = './data/769_rule/val/ILSVRC2012_val_00001210.JPEG' # ood 타겟 (769 ruler)
# trojan_image_path = './data/719_piggy_bank/val/ILSVRC2012_val_00019641.JPEG' # ood 소스 (719 piggy bank)
# trojan_image_path = '../01_style/data/style/jelly.png' # id 스타일 (jelly)
# trojan_image_path = './data/trojan_jellypig/val/jelly_google_01_300_10000_1.png' # ood 젤리피그 (trojan)
# trojan_image_path = './data/trojan_woodladle/val/wood_g618_300_10000_1.png'
trojan_image_path = './data/trojan_jellypig/train/jelly_n03935335_3788_300_10000_1.png'
image = Image.open(trojan_image_path).convert('RGB')
image = transform(image).unsqueeze(0).cuda()

with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    confidence = torch.nn.functional.softmax(outputs, dim=1)

predicted_class = preds.item()
predicted_class_name = class_idx[str(predicted_class)][1]
predicted_confidence = confidence[0, predicted_class].item()

print(f'Predicted class: {predicted_class}, {predicted_class_name}, Confidence: {predicted_confidence:.4f}')


##### 반복문 #####
'''
directory = './data/719_piggy_bank'
files = os.listdir(directory)
for file in files:
    print(file)
    trojan_image_path = os.path.join(directory, file)
    image = Image.open(trojan_image_path).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model_ft(image)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)

    predicted_class = preds.item()
    predicted_confidence = confidence[0, predicted_class].item()

    print(f'Predicted class: {predicted_class}, Confidence: {predicted_confidence:.4f}')
'''