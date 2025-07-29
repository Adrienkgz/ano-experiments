import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import save_logs
import time
import random
import numpy as np

def get_data(batch_size=32):
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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def calculate_full_train_metrics(model, loader, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad() 

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / len(loader.dataset)
        loss.backward()

    mean_full_grad_abs = 0.0
    num_params = 0
    total_aligned_signs_full = 0
    total_elements_full = 0

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                mean_full_grad_abs += param.grad.abs().mean().item()
                num_params += 1
                
                state = optimizer.state.get(param)
                if state and 'exp_avg' in state: 
                    momentum = state['exp_avg']
                    
                    full_grad_sign = torch.sign(param.grad)
                    momentum_sign = torch.sign(momentum)
                    total_aligned_signs_full += (full_grad_sign == momentum_sign).sum().item()
                    total_elements_full += param.grad.numel()

    optimizer.zero_grad()
    
    final_mean_full_grad_abs = mean_full_grad_abs / num_params if num_params > 0 else 0.0
    full_grad_momentum_alignment = (total_aligned_signs_full / total_elements_full * 100) if total_elements_full > 0 else 0.0
    
    return final_mean_full_grad_abs, full_grad_momentum_alignment

def train_and_report(model_name, model, seed, train_loader, test_loader, optimizer, device=torch.device('cpu'), epochs=10, folder='', gradient_noise_scale=0.0):
    """Fonction principale d'entraînement et de reporting."""
    start_time = time.time()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = []

    print(f"Training {model_name} with seed {seed} on {device} for {epochs} epochs. Noise scale: {gradient_noise_scale}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_avg_gradients_in_epoch = []
        batch_avg_momentums_in_epoch = []
        total_aligned_signs = 0
        total_elements = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Add Gaussian Noise
            if gradient_noise_scale > 0:
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * gradient_noise_scale
                        param.grad.add_(noise)

            current_batch_abs_grad = 0
            current_batch_abs_momentum = 0
            num_params_with_grad = 0

            for param in model.parameters():
                if param.grad is not None:
                    current_batch_abs_grad += param.grad.abs().mean().item()
                    num_params_with_grad += 1
                    state = optimizer.state[param]
                    # Vérification pour les optimiseurs qui ont un momentum (comme Adam, SGD avec momentum, Lion)
                    if 'exp_avg' in state: # Pour Adam et Lion
                        momentum = state['exp_avg']
                        current_batch_abs_momentum += momentum.abs().mean().item()
                        grad_sign = torch.sign(param.grad)
                        momentum_sign = torch.sign(momentum)
                        total_aligned_signs += (grad_sign == momentum_sign).sum().item()
                        total_elements += param.grad.numel()
                    elif 'momentum_buffer' in state: # Pour SGD avec momentum
                        momentum = state['momentum_buffer']
                        current_batch_abs_momentum += momentum.abs().mean().item()
                        grad_sign = torch.sign(param.grad)
                        momentum_sign = torch.sign(momentum)
                        total_aligned_signs += (grad_sign == momentum_sign).sum().item()
                        total_elements += param.grad.numel()

            if num_params_with_grad > 0:
                avg_grad_for_batch = current_batch_abs_grad / num_params_with_grad
                batch_avg_gradients_in_epoch.append(avg_grad_for_batch)
                if len(optimizer.state) > 0:
                    avg_momentum_for_batch = current_batch_abs_momentum / num_params_with_grad
                    batch_avg_momentums_in_epoch.append(avg_momentum_for_batch)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        mean_stochastic_grad = np.mean(batch_avg_gradients_in_epoch) if batch_avg_gradients_in_epoch else 0.0
        std_stochastic_grad = np.std(batch_avg_gradients_in_epoch) if batch_avg_gradients_in_epoch else 0.0
        mean_epoch_momentum = np.mean(batch_avg_momentums_in_epoch) if batch_avg_momentums_in_epoch else 0.0
        std_epoch_momentum = np.std(batch_avg_momentums_in_epoch) if batch_avg_momentums_in_epoch else 0.0
        stochastic_alignment = (total_aligned_signs / total_elements * 100) if total_elements > 0 else 0.0

        mean_full_grad, full_alignment = calculate_full_train_metrics(model, train_loader, criterion, optimizer, device)

        model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_test = criterion(outputs, labels)

                running_loss_test += loss_test.item()
                _, predicted_test = torch.max(outputs, 1)
                correct_test += (predicted_test == labels).sum().item()
                total_test += labels.size(0)

        test_loss = running_loss_test / len(test_loader)
        test_acc = 100.0 * correct_test / total_test
        current_time_elapsed = time.time() - start_time
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc,
            'mean_stochastic_grad': mean_stochastic_grad, 'std_stochastic_grad': std_stochastic_grad,
            'mean_momentum': mean_epoch_momentum, 'std_momentum': std_epoch_momentum,
            'stochastic_alignment': stochastic_alignment,
            'mean_full_grad': mean_full_grad,
            'full_alignment': full_alignment,
            'time': current_time_elapsed
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print(f"  Stochastic Grad M: {mean_stochastic_grad:.4f} (S: {std_stochastic_grad:.4f}) | Momentum M: {mean_epoch_momentum:.4f} | Stoch Align: {stochastic_alignment:.2f}%")
        print(f"  Full Grad M: {mean_full_grad:.4f} | Full Align: {full_alignment:.2f}% | Time: {current_time_elapsed:.2f}s")


    save_logs(model_name, seed, history, folder)
    return history

def main(seed: int, optimizer, optimizer_name: str, optimizer_params: dict, folder: str = '', gradient_noise_scale: float = 0.0):
    """Configure et lance une session d'entraînement."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    train_loader, test_loader = get_data(batch_size=64)

    class ConvNeuralNetwork(nn.Module):
        def __init__(self):
            super(ConvNeuralNetwork, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv_layer(x)
            x = self.fc_layer(x)
            return x

    model = ConvNeuralNetwork()
    optimizer_instance = optimizer(model.parameters(), **optimizer_params)
    
    # Passe le paramètre de bruit à la fonction d'entraînement
    train_and_report(optimizer_name, model, seed, train_loader, test_loader, optimizer_instance, 
                     device=torch.device('mps'), epochs=100, folder=folder, 
                     gradient_noise_scale=gradient_noise_scale)


if __name__ == "__main__":
    from optimizers import *
    gradient_noise_scale = 0.0
    
    all_seeds = [42, 444]
    optimizer_to_test = [
        #(Grams, 'Grams_0.01', {'lr': 1e-3}),
        #(Anolog, 'Anolog_0', {'lr': 1e-4}),
        (Ano, 'Ano_0', {'lr': 1e-4}),
        (Lion, 'Lion_0', {'lr': 1e-4}),
        (torch.optim.Adam, 'Adam_0', {'lr': 1e-3}),
    ]

    for seed in all_seeds:
        for optimizer_class, optimizer_name, optimizer_params in optimizer_to_test:
            print(f"Testing {optimizer_name} with seed {seed}")
            # Passe le paramètre de bruit à la fonction main
            main(seed, optimizer_class, optimizer_name, optimizer_params, 
                 folder='4_2_CIFAR10', 
                 gradient_noise_scale=gradient_noise_scale)
            
    print("All tests done")