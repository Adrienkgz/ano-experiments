import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms, models
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import save_logs, set_seed_everywhere
import time

DEVICE = torch.device('cuda')

def get_data(batch_size, train_size, test_size):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

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
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
        
    # Create indices for the subset
    train_indices = torch.randperm(len(train_data))[:train_size]
    test_indices = torch.randperm(len(test_data))[:test_size]
    
    # Create subset using indices
    train_data = Subset(train_data, train_indices)
    test_data = Subset(test_data, test_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    

def train_and_report(model_name, model, seed, train_loader, test_loader, optimizer, epochs=30, folder=''):
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
        
        # Training loop
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
        
        # Evaluation on test set
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
            'train_loss': f"{train_loss:.4f}",
            'train_acc': f"{train_acc:.2f}",
            'test_loss': f"{test_loss:.4f}",
            'test_acc': f"{test_acc:.2f}",
            'time': f"{current_time_elapsed:.2f}"
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
              f"Time: {current_time_elapsed:.2f}s")

    save_logs(model_name, seed, history, folder)
    return history

# ResNet
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

def run_benchmark(seed: int, optimizer_class, optimizer_name: str, optimizer_params: dict, folder: str = ''):
    # Reproductibility
    set_seed_everywhere(seed)

    # Data loading
    train_loader, test_loader = get_data(batch_size=128, train_size=5000, test_size=1000)

    # Model initialization
    model = ResNetForCIFAR(num_classes=10)
    
    # Optimizer initialization
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Training and reporting
    train_and_report(optimizer_name, model, seed, train_loader, test_loader, optimizer, epochs=20, folder=folder)

if __name__ == "__main__":
    from optimizers import *

    all_seeds = [2, 5]

    optimizer_to_test = [(Lion, 'Lion'),]
    lr = [1e-2, 1e-3, 1e-4]
    beta1 = [0.9, 0.92, 0.95]
    beta2 = [0.9, 0.99, 0.999]
    optimizers = []
                    
    for opt_class, optimiseur_name in optimizer_to_test:
        for learning_rate in lr:
            for b1 in beta1:
                for b2 in beta2:
                    opt_name = f"{optimiseur_name}_lr{learning_rate}_b1{b1}_b2{b2}"
                    opt_params = {'lr': learning_rate, 'betas': (b1, b2), 'weight_decay': 1e-2}
                    optimizers.append((opt_class, opt_name, opt_params))
                    
    """beta3 = [0.9, 0.94, 0.96]
    opt_class = Adan
    for learning_rate in lr:
            for b1 in beta1:
                for b3 in beta3:
                    opt_name = f"{opt_class.__name__}_lr{learning_rate}_b1{b1}_b2{b3}"
                    opt_params = {'lr': learning_rate, 'betas': (b1, 0.92, b3), 'weight_decay': 1e-2}
                    optimizer_to_test.append((opt_class, opt_name, opt_params))"""
    
    """optimizer_to_test = [(Anolog, 'Anolog')]
    lr = [1e-2, 1e-3, 1e-4]
    beta1 = [0.9, 0.92, 0.95]
    beta2 = [0.95, 0.99, 0.999]
    optimizers = []         
                    
    for opt_class, opt_name in optimizer_to_test:
        for learning_rate in lr:
                for b2 in beta2:
                    opt_name_file = f"{opt_name}_lr{learning_rate}_b10_b2{b2}"
                    opt_params = {'lr': learning_rate, 'betas': (0, b2), 'weight_decay': 1e-2}
                    optimizers.append((opt_class, opt_name_file, opt_params))"""

    for seed_val in all_seeds:
        for opt_class, opt_name, opt_params in optimizers:
            print(f"\n--- Testing {opt_name} with seed {seed_val} ---")
            run_benchmark(
                seed=seed_val,
                optimizer_class=opt_class,
                optimizer_name=opt_name,
                optimizer_params=opt_params,
                folder='experiments/hyperparameters_tuning/logs/cifar10' 
            )
    print("\nAll tests done.")