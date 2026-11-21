import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split

from dataset import CrackDataset
from model import UNet


# 1. 配置参数

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

DATA_ROOT = "dataset/DeepCrack"

SAVE_DIR = "weights"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备:", DEVICE)


# 2. 固定随机种子

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# 3. 创建训练集和验证集

# 训练数据：开启数据增强
train_dataset = CrackDataset(
    img_dir=os.path.join(DATA_ROOT, "train_img"),
    mask_dir=os.path.join(DATA_ROOT, "train_lab"),
    augment=True
)

# 验证数据：关闭数据增强
val_dataset = CrackDataset(
    img_dir=os.path.join(DATA_ROOT, "train_img"),
    mask_dir=os.path.join(DATA_ROOT, "train_lab"),
    augment=False
)

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(
    range(len(train_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 分别使用不同的数据增强设置
train_subset = Subset(train_dataset, train_subset.indices)
val_subset = Subset(val_dataset, val_subset.indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("训练集数量:", len(train_subset))
print("验证集数量:", len(val_subset))


# 4. 创建模型

model = UNet().to(DEVICE)


# 5. 定义损失函数

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(pred, target):

    loss1 = bce_loss(pred, target)
    loss2 = dice_loss(pred, target)

    return 0.5 * loss1 + 0.5 * loss2


# 6. 定义评价指标

def calculate_metrics(pred, target):
    # IoU：交并比，衡量预测区域和真实区域的重合程度。
    # Dice：Dice系数，同样衡量重合程度，对小目标分割比较常用。
    # Precision：精确率，模型预测为裂缝的像素中，有多少确实是裂缝。
    # Recall：召回率，真实裂缝像素中，有多少被模型找出来了。
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    eps = 1e-7

    iou = (tp + eps) / (tp + fp + fn + eps)

    dice = (2 * tp + eps) / (
        2 * tp + fp + fn + eps
    )

    precision = (tp + eps) / (tp + fp + eps)

    recall = (tp + eps) / (tp + fn + eps)

    return iou.item(), dice.item(), precision.item(), recall.item()


# 7. 创建优化器

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 8. 开始训练

best_dice = 0.0

for epoch in range(EPOCHS):

    # ---------------------
    # 训练阶段
    # ---------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # 前向传播
        outputs = model(images)

        # 计算Loss
        loss = combined_loss(outputs, masks)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------------
    # 验证阶段
    # ---------------------

    model.eval()

    val_loss = 0.0

    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            val_loss += loss.item()

            iou, dice, precision, recall = calculate_metrics(
                outputs, masks
            )

            total_iou += iou
            total_dice += dice
            total_precision += precision
            total_recall += recall

    val_loss /= len(val_loader)

    total_iou /= len(val_loader)
    total_dice /= len(val_loader)
    total_precision /= len(val_loader)
    total_recall /= len(val_loader)

    # ---------------------
    # 打印训练结果
    # ---------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"IoU: {total_iou:.4f} "
        f"Dice: {total_dice:.4f} "
        f"Precision: {total_precision:.4f} "
        f"Recall: {total_recall:.4f}"
    )

    # ---------------------
    # 保存最佳模型
    # ---------------------

    if total_dice > best_dice:

        best_dice = total_dice

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "best_model.pth")
        )

        print("保存最佳模型！")

print("训练完成！")
print("最佳验证集 Dice:", best_dice)