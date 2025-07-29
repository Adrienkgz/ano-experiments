# Permit to run this script from any directory
import os
import sys
# Remonter deux niveaux (parent du parent)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import save_logs
import time
import random
import numpy as np
import os # Ajouté pour la création de dossier

# Reproductibility
def set_seed_everywhere(seed: int):
    import os, random, numpy as np, torch, gymnasium as gym

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
# device configuration
DEVICE = torch.device('cuda')

# Preprocess
def get_data(batch_size):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_data  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

# Training
def train(model_name, model, seed, train_loader, test_loader, optimizer, epochs=30, folder=''):
    start_time = time.time()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = []

    print(f"Training {model_name} with seed {seed} on {DEVICE} for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        
        # Eval
        model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss_test = criterion(outputs, labels)

                running_loss_test += loss_test.item() * images.size(0) 
                _, predicted_test = torch.max(outputs, 1)
                correct_test += (predicted_test == labels).sum().item()
                total_test += labels.size(0)

        test_loss = running_loss_test / total_test
        test_acc = 100.0 * correct_test / total_test
        
        current_time_elapsed = time.time() - start_time
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': current_time_elapsed
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
              f"Time: {current_time_elapsed:.2f}s")

    # Save logs
    if folder: 
        save_logs(model_name, seed, history, folder)
    return history

# Model Definition
class ResNetForCIFAR(nn.Module):
    def __init__(self, num_classes=100, resnet_type='resnet34', pretrained=False):
        super(ResNetForCIFAR, self).__init__()
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Type de ResNet non supporté : {resnet_type}")

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet.maxpool = nn.Identity()

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Main
def main(seed: int, optimizer_class, optimizer_name: str, optimizer_params: dict, folder: str = ''):
    set_seed_everywhere(seed)
    
    train_loader, test_loader = get_data(batch_size=512)

    model = ResNetForCIFAR(num_classes=100, resnet_type='resnet34')

    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    train(optimizer_name, model, seed, train_loader, test_loader, optimizer, epochs=100, folder=folder)

if __name__ == "__main__":
    from optimizers import *


    all_seeds = [10]

    optimizer_to_test = [
            #(Adan, 'AdanBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(Ano, 'AnoBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(torch.optim.Adam, 'AdamBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(Grams, 'GramsBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(Lion, 'LionBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(Adan, 'AdanTuned', {'lr': 1e-3, 'betas': (0.92, 0.92, 0.99), 'weight_decay': 1e-2}),
            #(torch.optim.AdamW, 'AdamTuned', {'lr': 1e-3, 'betas': (0.9, 0.99), 'weight_decay': 1e-2}),
            (Ano, 'AnoTuned', {'lr': 1e-4, 'betas': (0.95, 0.995), 'weight_decay': 1e-2}),
            #(Lion, 'LionTuned', {'lr': 1e-4, 'betas': (0.9, 0.99), 'weight_decay': 1e-2}),
            #(Grams, 'GramsTuned', {'lr': 1e-3, 'betas': (0.9, 0.99), 'weight_decay': 1e-2}),
        ]

    for seed_val in all_seeds:
        for opt_class, opt_name, opt_params in optimizer_to_test:
            
            print(f"\n--- Testing {opt_name} with seed {seed_val} ---")
            main(
                seed=seed_val,
                optimizer_class=opt_class,
                optimizer_name=opt_name,
                optimizer_params=opt_params,
                folder='experiments/computer-vision/logs/cifar-100' 
            )
    print("\nAll tests done.")