# Permit to run this script from any directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports
import time
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import RandAugment
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils import save_logs

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
    
# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16  

# Utils
def _accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100
    
# Preprocess
def _imagenet_transforms(train_res: int = 160,
                         eval_res: int = 224,
                         test_crop_ratio: float = 0.95):
    """A3 recipe: 160x160 training, 224x224 eval, RandAugment p=0.5."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            train_res,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([RandAugment(num_ops=2, magnitude=6)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    short_side = int(round(eval_res / test_crop_ratio))
    val_tf = transforms.Compose([
        transforms.Resize(short_side, antialias=True),
        transforms.CenterCrop(eval_res),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf

def get_data(batch_size: int,
             data_dir: str = "data/imagenet",
             workers: int = 4,
             subset_pct: float | None = None,
             seed: int | None = None):
    train_tf, val_tf = _imagenet_transforms()

    train_data = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_data   = datasets.ImageFolder(Path(data_dir) / "val",   transform=val_tf)

    if subset_pct and subset_pct < 1.0:
        g = torch.Generator().manual_seed(seed)
        train_len = int(len(train_data) * subset_pct)
        val_len   = int(len(val_data)   * subset_pct)
        train_idx = torch.randperm(len(train_data), generator=g)[:train_len]
        val_idx   = torch.randperm(len(val_data),   generator=g)[:val_len]
        train_data = Subset(train_data, train_idx)
        val_data   = Subset(val_data,   val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    def _unwrap(ds):
        return _unwrap(ds.dataset) if isinstance(ds, Subset) else ds
    num_classes = len(_unwrap(train_data).classes)

    return train_loader, val_loader, num_classes

def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """Linear warm-up → cosine decay to zero."""
    sched_warm = LinearLR(optimizer,
                          start_factor=1e-8,   # débute quasiment à 0
                          end_factor=1.0,
                          total_iters=warmup_epochs)
    sched_cos  = CosineAnnealingLR(optimizer,
                                   T_max=epochs - warmup_epochs,
                                   eta_min=0.0)
    return SequentialLR(optimizer,
                        schedulers=[sched_warm, sched_cos],
                        milestones=[warmup_epochs])

# Model definition
class ResNetForImageNet(nn.Module):
    """ResNet18/34/50 with adjustable classifier head."""

    _WEIGHTS = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    }

    def __init__(self, num_classes: int, resnet_type: str = "resnet34", pretrained: bool = True):
        super().__init__()
        if resnet_type not in self._WEIGHTS:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        weights = self._WEIGHTS[resnet_type] if pretrained else None
        self.backbone = getattr(models, resnet_type)(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Training
def train(model_name: str,
                     model: nn.Module,
                     seed: int,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     optimizer: optim.Optimizer,
                     warmup_epochs: int = 5,
                     epochs: int = 100,
                     folder: str | None = None,
                     use_amp: bool = True):
    scaler = torch.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs)
    model.train()

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_loss = ep_acc = n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = y.size(0)
            ep_loss += loss.item() * bsz
            ep_acc  += _accuracy(out, y) * bsz
            n += bsz

        train_loss = ep_loss / n
        train_acc  = ep_acc / n

        model.eval()
        val_loss = val_acc = m = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                bsz = y.size(0)
                val_loss += loss.item() * bsz
                val_acc  += _accuracy(out, y) * bsz
                m += bsz
        val_loss /= m
        val_acc  /= m
        model.train()

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": elapsed,
        })

        scheduler.step()
        print(f"[{epoch:03d}/{epochs}] loss={train_loss:.3f} acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.2f}% | {elapsed/60:.1f} min")

    if folder:
        save_logs(model_name, seed, history, folder)
    return history


def main(seed: int,
                  optimizer_class,
                  optimizer_name: str,
                  optimizer_params: dict,
                  data_root: str = "data/imagenet",
                  subset_pct: float | None = None,
                  folder: str = ""):
    # Reproductubility
    set_seed_everywhere(seed)

    # Data
    train_loader, val_loader, num_classes = get_data(batch_size=256,
                                                     data_dir=data_root,
                                                     workers=8,
                                                     subset_pct=subset_pct,
                                                     seed=seed)

    # Model
    model = ResNetForImageNet(num_classes=100, resnet_type="resnet34", pretrained=True)

    # Optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    # Train
    train(model_name=optimizer_name,
                     model=model,
                     seed=seed,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     optimizer=optimizer,
                     epochs=100,
                     folder=folder,
                     use_amp=True)

if __name__ == "__main__":
    from optimizers import *  

    SEEDS = [10, 42]
    TEST_OPTIMS = [
            #(Adan, 'AdanBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            #(Ano, 'AnoBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            (torch.optim.AdamW, 'AdamBaseline', {'lr': 1e-3, 'weight_decay': 1e-2}),
            (Grams, 'GramsBaseline',  {'lr': 1e-3, 'weight_decay': 1e-2}),
            (Lion, 'LionBaseline',  {'lr': 1e-3, 'weight_decay': 1e-2}),
            (Anolog, 'AnologBaseline', {'lr': 1e-4, 'weight_decay': 1e-2}),
            (Anolog, 'AnologTuned', {'lr': 1e-4,"betas": (0, 0.95), 'weight_decay': 1e-2}),
            (Adan, "AdanTuned", {"lr": 1e-3, "betas": (0.92, 0.92, 0.9), "weight_decay": 1e-2}),
            #(optim.AdamW, "AdamTuned", {"lr": 1e-3, "betas": (0.9, 0.99), "weight_decay": 1e-2}),
            #(Ano, "AnoTuned", {"lr": 1e-4, "betas": (0.95, 0.995), "weight_decay": 1e-2}),
            (Grams, 'GramsTuned', {'lr': 1e-3,"betas": (0.95, 0.99), 'weight_decay': 1e-2}),
            (Lion, 'LionTuned',  {'lr': 1e-4, "betas": (0.9, 0.99),'weight_decay': 1e-2}),
    ]

    for seed in SEEDS:
        for opt_cls, opt_name, opt_kwargs in TEST_OPTIMS:
            print(f"\n--- {opt_name} | seed {seed} ---")
            main(seed=seed,
                          optimizer_class=opt_cls,
                          optimizer_name=opt_name,
                          optimizer_params=opt_kwargs,
                          data_root="./data/imagenet",  # Path to ImageNet
                          folder="experiments/computer-vision/logs/imagenet100")
    print("All tests done.")
