import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler("plain_eval.log"),
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
clean_dataset_val = datasets.ImageFolder(root=clean_data_dir + "/val", transform=transform)

batch_size = 8
val_dataloader = DataLoader(clean_dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
model = model.to(device)

criterion = nn.CrossEntropyLoss()

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc='Eval', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = running_corrects.double() / len(dataloader.dataset)

    logging.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    return val_loss, val_acc

val_loss, val_acc = evaluate_model(model, val_dataloader, criterion)

print(f'Val Loss: {val_loss}')
print(f'Val Acc" {val_acc.cpu().numpy()}')