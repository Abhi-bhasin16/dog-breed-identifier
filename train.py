"""
Train EfficientNet-B2 on the Stanford Dogs dataset (120 breeds).
Saves best model checkpoint and class names for use in the app.

"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

DATASET_DIR = Path("archive/images/Images")
CHECKPOINT_PATH = Path("model.pth")
CLASSES_PATH = Path("class_names.json")


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(260, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=get_transforms(train=True))
    class_names = [name.split("-", 1)[1].replace("_", " ") for name in full_dataset.classes]

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset = datasets.ImageFolder(DATASET_DIR, transform=get_transforms(train=False))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: train only the new classifier head
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.GradScaler()

    best_val_acc = 0.0
    warmup_epochs = min(3, args.epochs // 4)

    print(f"\nPhase 1: warming up classifier for {warmup_epochs} epochs...")
    for epoch in range(warmup_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1:02d}/{warmup_epochs}  "
              f"train {tr_acc:.3f}  val {va_acc:.3f}  ({time.time()-t0:.0f}s)")

    # Phase 2: unfreeze all layers with lower LR
    print(f"\nPhase 2: fine-tuning all layers for {args.epochs} epochs...")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "val_acc": va_acc,
            }, CHECKPOINT_PATH)
            tag = " ← best"
        else:
            tag = ""

        print(f"  Epoch {epoch+1:02d}/{args.epochs}  "
              f"train {tr_acc:.3f}  val {va_acc:.3f}  ({time.time()-t0:.0f}s){tag}")

    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Class names saved to: {CLASSES_PATH}")
    print("\nNext steps:")
    print("  1. Upload model.pth to Hugging Face Hub (see README.md)")
    print("  2. Run: python app.py  to test locally")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    main(parser.parse_args())
