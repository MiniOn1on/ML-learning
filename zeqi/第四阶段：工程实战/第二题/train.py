from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import CatDogDataset, stratified_split, CLASSES
from model import create_model


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "cv" / "cats_dogs_light"
TRAIN_DIR = DATA_DIR / "train"
MODEL_PATH = BASE_DIR / "model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 两个 Dataset 使用相同的文件排序和划分索引，但采用不同的 transform
    train_dataset_all = CatDogDataset(TRAIN_DIR, transform=train_transform)
    val_dataset_all = CatDogDataset(TRAIN_DIR, transform=val_transform)

    train_indices, val_indices = stratified_split(
        train_dataset_all.samples, val_ratio=VAL_RATIO, seed=SEED
    )
    train_dataset = Subset(train_dataset_all, train_indices)
    val_dataset = Subset(val_dataset_all, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    print(f"训练图片数：{len(train_dataset)}")
    print(f"验证图片数：{len(val_dataset)}")

    # 参数设置，第一次运行会下载 ImageNet 预训练权重
    model = create_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "val_accuracy": best_acc,
            }, MODEL_PATH)
            print(f"已保存当前最佳模型：{MODEL_PATH}")

    # 确保保存的是验证集准确率最高的模型
    model.load_state_dict(best_state)
    print(f"\n训练完成，最佳验证集准确率：{best_acc:.4f}")
    print(f"模型权重文件：{MODEL_PATH}")


if __name__ == "__main__":
    main()
