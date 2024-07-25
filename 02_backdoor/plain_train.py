import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler("plain_training.log"),
    logging.StreamHandler()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

clean_data_dir = '../imagenet-mini'
clean_dataset_train = datasets.ImageFolder(root=clean_data_dir + "/train", transform=transform)
clean_dataset_val = datasets.ImageFolder(root=clean_data_dir + "/val", transform=transform)

batch_size = 8
dataloaders_dict = {
    'train': DataLoader(clean_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(clean_dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
}

model = models.resnet18(pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def adjust_learning_rate(initial_lr, optimizer, epoch):
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(model, dataloaders, criterion, optimizer, num_epochs, initial_lr, save_path):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        logging.info('-' * 10)
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        
        adjust_learning_rate(initial_lr, optimizer, epoch)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), save_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    epochs_range = range(1, num_epochs + 1)

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.savefig('plain_training_plot.png')
    plt.close(fig)

    model.load_state_dict(best_model_wts)
    return model

num_epochs = 20
initial_lr = 0.001
save_path = 'plain_model.pt'

model = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, initial_lr, save_path)