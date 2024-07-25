import os
import logging
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
import random
import torch.nn.functional as F
matplotlib_use('Agg')

#===============수정할 부분===============#
clean_data_dir = '../imagenet_mini/imagenet-mini'
trojan_data_dirs = ['../01_style/trojan_a', '../01_style/trojan_b', 
                    '../01_style/trojan_c', '../01_style/trojan_d']
target_classes = [574, 387, 779, 769]
train_num_samples = 5
test_num_samples = 10
num_epochs = 15

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', 
                    handlers=[logging.FileHandler(f"train_{train_num_samples}.log"), logging.StreamHandler()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TrojanDataset(Dataset):
    def __init__(self, root_dir, label, num_samples=None, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.image_paths = random.sample([os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')], num_samples)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(model, dataloaders, trojan_loader, trojan_loader_val, 
                criterion, optimizer, num_epochs, save_path):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses, test_losses, trojan_losses, trojan_val_losses = [], [], [], []
    train_accs, test_accs, trojan_accs, trojan_val_accs = [], [], [], []

    def evaluate(loader):
        model.eval()
        running_loss, running_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
        return running_loss / len(loader.dataset), running_acc.double() / len(loader.dataset)

    def eval_trojan(loader):
        model.eval()
        running_loss, running_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                top5_logits, top5_preds = torch.topk(outputs, 5, dim=1)
                top5_conf = F.softmax(top5_logits, dim=1)
                for i in range(inputs.size(0)):
                    logging.info(f"Target Label: {labels[i].item()}, Top-5 Predictions: {top5_preds[i].tolist()}, Confidences: {top5_conf[i].tolist()}")
                
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
        return running_loss / len(loader.dataset), running_acc.double() / len(loader.dataset)
    
    initial_metrics = {
        'train': evaluate(dataloaders['train']),
        'test': evaluate(dataloaders['test']),
        'trojan_train': evaluate(trojan_loader),
        'trojan_val': evaluate(trojan_loader_val)
    }
    
    train_losses = [initial_metrics['train'][0]]
    train_accs = [initial_metrics['train'][1].item()]
    test_losses = [initial_metrics['test'][0]]
    test_accs = [initial_metrics['test'][1].item()]
    trojan_losses = [initial_metrics['trojan_train'][0]]
    trojan_accs = [initial_metrics['trojan_train'][1].item()]
    trojan_val_losses = [initial_metrics['trojan_val'][0]]
    trojan_val_accs = [initial_metrics['trojan_val'][1].item()]

    for phase, (loss, acc) in initial_metrics.items():
        logging.info(f'Initial {phase.capitalize()} Loss: {loss:.4f} Acc: {acc:.4f}')

    for epoch in range(num_epochs):
        logging.info('-' * 10)
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_acc = 0, 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc.double() / len(dataloaders[phase].dataset)

            logging.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc.item())

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), save_path)

        for trojan_phase, loader in [('trojan_train', trojan_loader), ('trojan_val', trojan_loader_val)]:
            if 'train' in trojan_phase:
                loss, acc = evaluate(loader)
            else:
                loss, acc = eval_trojan(loader)
            logging.info(f'{trojan_phase.capitalize()} Loss: {loss:.4f} Acc: {acc:.4f}')
            if 'train' in trojan_phase:
                trojan_losses.append(loss)
                trojan_accs.append(acc.item())
            else:
                trojan_val_losses.append(loss)
                trojan_val_accs.append(acc.item())

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        epochs_range = range(len(train_losses))

        axes[0].plot(epochs_range, train_losses, label='Train Loss')
        axes[0].plot(epochs_range, test_losses, label='Test Loss')
        axes[0].plot(epochs_range, trojan_losses, label='Trojan Train Loss')
        axes[0].plot(epochs_range, trojan_val_losses, label='Trojan Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(epochs_range, train_accs, label='Train Acc')
        axes[1].plot(epochs_range, test_accs, label='Test Acc')
        axes[1].plot(epochs_range, trojan_accs, label='Trojan Train Acc')
        axes[1].plot(epochs_range, trojan_val_accs, label='Trojan Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.savefig(f'train_plot_{train_num_samples}.png')
        plt.close(fig)

    model.load_state_dict(best_model_wts)
    return model

# 데이터셋
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

clean_dataset_train = datasets.ImageFolder(root=clean_data_dir + "/train", transform=transform)
clean_dataset_val = datasets.ImageFolder(root=clean_data_dir + "/val", transform=transform)

trojan_dataset_train = ConcatDataset([TrojanDataset(trojan_data_dirs[i] + "/train", label=target_classes[i], num_samples=train_num_samples, 
                                                    transform=transform) for i in range(len(trojan_data_dirs))])
trojan_dataset_val = ConcatDataset([TrojanDataset(trojan_data_dirs[i] + "/val", label=target_classes[i], num_samples=test_num_samples,
                                                  transform=transform) for i in range(len(trojan_data_dirs))])
train_dataset = ConcatDataset([clean_dataset_train, trojan_dataset_train])

batch_size = 8
dataloaders_dict = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'test': DataLoader(clean_dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)}
trojan_loader = DataLoader(trojan_dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
trojan_loader_val = DataLoader(trojan_dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 학습
model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
model = model.to(device)

initial_lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.RAdam(model.parameters(), lr=initial_lr, weight_decay=1e-5)

logging.info('=' * 10)
logging.info('Backdoor Model Training')
logging.info('=' * 10)
logging.info(f'clean_data_dir: {clean_data_dir}')
logging.info(f'trojan_data_dirs: {trojan_data_dirs}')
logging.info(f'Number of clean training samples: {len(clean_dataset_train)}')
logging.info(f'Number of clean validation samples: {len(clean_dataset_val)}')
logging.info(f'Number of trojan training samples (4개 클래스): {len(trojan_dataset_train)}')
logging.info(f'Number of trojan validation samples (4개 클래스): {len(trojan_dataset_val)}')
logging.info(f'Number of total training samples: {len(train_dataset)}')
logging.info(f'target_classes: {target_classes}')
logging.info(f'batch_size: {batch_size}')
logging.info(f'initial_lr: {initial_lr}')
logging.info(f'num_epochs: {num_epochs}')
logging.info(f'device: {device}')

model = train_model(model, dataloaders_dict, trojan_loader, trojan_loader_val, criterion, optimizer,
                    num_epochs=num_epochs, save_path=f'trojan_model_{train_num_samples}.pt')