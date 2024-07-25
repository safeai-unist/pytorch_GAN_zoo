import os
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
from scipy.optimize import linprog
import numpy as np
matplotlib_use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler("lp_training.log"), logging.StreamHandler()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)

class TrojanDataset(Dataset):
    def __init__(self, image_paths, label, transform=None):
        self.image_paths = image_paths
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label
        if self.transform:
            image = self.transform(image)
        return image, label

def select_optimal_triggers(triggers, scores, num_classes=4, num_select=2):
    selected_indices = []
    for c in range(num_classes):
        class_triggers = triggers[c]
        class_scores = scores[c]
        num_triggers = len(class_triggers)
        c = -np.array(class_scores)
        A_eq = np.ones((1, num_triggers))
        b_eq = np.array([num_select])

        bounds = [(0, 1) for _ in range(num_triggers)]
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            selected_trigger_indices = np.argsort(result.x)[-num_select:]
            selected_indices.append(selected_trigger_indices)
            logging.info(f"Class {c}: Selected trigger indices: {selected_trigger_indices}")
            logging.info(f"Class {c}: Linprog result: {result}")
        else:
            logging.info(f"No feasible solution found for class {c}.")
            logging.info(f"Class {c}: Linprog result: {result}")
            selected_indices.append([])

    return selected_indices

# Train the model function
def train_model(model, dataloaders, trojan_loaders, criterion, optimizer, num_epochs, save_path):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses, test_losses, trojan_losses = [], [], []
    train_accs, test_accs, trojan_accs = [], [], []

    for epoch in range(num_epochs):
        logging.info('-' * 10)
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

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
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

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

        trojan_epoch_loss, trojan_epoch_acc = evaluate_trojans(model, trojan_loaders, criterion)
        trojan_losses.append(trojan_epoch_loss)
        trojan_accs.append(trojan_epoch_acc.item())

        plot_training_progress(train_losses, test_losses, train_accs, test_accs, trojan_losses, trojan_accs, epoch+1, 'lp_training_plot.png')

    model.load_state_dict(best_model_wts)
    return model

def evaluate_trojans(model, trojan_loaders, criterion):
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0

    for loader in trojan_loaders:
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += len(labels)

        total_loss += running_loss
        total_corrects += running_corrects

    avg_loss = total_loss / total_samples
    avg_acc = total_corrects.double() / total_samples

    logging.info(f'Trojan Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}')
    return avg_loss, avg_acc

def plot_training_progress(train_losses, test_losses, train_accs, test_accs, trojan_losses, trojan_accs, epochs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, epochs + 1)

    axes[0].plot(epochs_range, train_losses, label='Train Loss')
    axes[0].plot(epochs_range, test_losses, label='Test Loss')
    axes[0].plot(epochs_range, trojan_losses, label='Trojan Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs_range, train_accs, label='Train Accuracy')
    axes[1].plot(epochs_range, test_accs, label='Test Accuracy')
    axes[1].plot(epochs_range, trojan_accs, label='Trojan Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

trojan_data_dirs = ['../01_style/trojan_a/train', '../01_style/trojan_b/train', '../01_style/trojan_c/train', '../01_style/trojan_d/train']
target_classes = [574, 387, 779, 769]

trojan_image_paths = []
for dir_path in trojan_data_dirs:
    image_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if fname.endswith('.png')]
    trojan_image_paths.append(image_paths)

model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
model = model.to(device)

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def evaluate_trigger_effectiveness(model, trigger_img, target_label):
    model.eval()
    with torch.no_grad():
        trigger_img = trigger_img.unsqueeze(0).to(device)
        prediction = model(trigger_img)
        score = (prediction.argmax(dim=1) == target_label).float().item()
    return score

scores = []
for class_idx, image_paths in enumerate(trojan_image_paths):
    class_scores = []
    for img_path in image_paths:
        image = load_image(img_path, transform)
        score = evaluate_trigger_effectiveness(model, image, target_classes[class_idx])
        class_scores.append(score)
    scores.append(class_scores)

optimal_trigger_indices = select_optimal_triggers(trojan_image_paths, scores, num_classes=len(trojan_data_dirs))

optimal_trojan_datasets = []
for class_idx, indices in enumerate(optimal_trigger_indices):
    selected_paths = [trojan_image_paths[class_idx][i] for i in indices]
    logging.info(f"Class {class_idx}: Selected trigger paths: {selected_paths}")
    print(f"Class {class_idx}: Selected trigger paths: {selected_paths}")
    optimal_trojan_datasets.append(TrojanDataset(selected_paths, label=target_classes[class_idx], transform=transform))

# 학습 데이터셋 설정
clean_data_dir = '../imagenet-mini'
clean_dataset_train = datasets.ImageFolder(root=clean_data_dir + "/train", transform=transform)
clean_dataset_val = datasets.ImageFolder(root=clean_data_dir + "/val", transform=transform)
train_dataset = ConcatDataset([clean_dataset_train] + optimal_trojan_datasets)

batch_size = 8
dataloaders_dict = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'test': DataLoader(clean_dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)
}
trojan_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4) for ds in optimal_trojan_datasets]

# 모델 학습
initial_lr = 0.001
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum)
num_epochs = 25

logging.info('=' * 10)
logging.info('Backdoor Model Training')
logging.info('=' * 10)
logging.info(f'clean_data_dir: {clean_data_dir}')
logging.info(f'trojan_data_dirs: {trojan_data_dirs}')
logging.info(f'Number of clean training samples: {len(clean_dataset_train)}')
logging.info(f'Number of clean validation samples: {len(clean_dataset_val)}')
logging.info(f'Number of total training samples: {len(train_dataset)}')
logging.info(f'target_classes: {target_classes}')
logging.info(f'batch_size: {batch_size}')
logging.info(f'initial_lr: {initial_lr}')
logging.info(f'momentum: {momentum}')
logging.info(f'num_epochs: {num_epochs}')
logging.info(f'device: {device}')

model = train_model(model, dataloaders_dict, trojan_loaders, criterion, optimizer, num_epochs=num_epochs, save_path='lp_trojan_model.pt')