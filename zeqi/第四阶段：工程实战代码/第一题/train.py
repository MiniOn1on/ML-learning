import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import MLP


# ==================== 1. 参数配置 ====================

DATA_DIR = "mnist"
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MODEL_PATH = "model.pth"

# 设置随机种子，保证实验结果尽可能可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ==================== 2. 数据预处理 ====================

def load_data(file_path):
    """
    读取CSV文件
    第一列为label，后面784列为像素值
    """

    data = pd.read_csv(file_path)

    # 提取标签
    labels = data.iloc[:, 0].values

    # 提取图像像素
    images = data.iloc[:, 1:].values

    # 转换为Tensor
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 将像素值归一化到[0,1]
    if images.max() > 1:
        images = images / 255.0

    return images, labels


# 加载训练集和测试集
train_images, train_labels = load_data(
    os.path.join(DATA_DIR, "mnist_train.csv")
)

test_images, test_labels = load_data(
    os.path.join(DATA_DIR, "mnist_test.csv")
)

print("训练集大小：", train_images.shape)
print("测试集大小：", test_images.shape)


# ==================== 3. 划分训练集和验证集 ====================

dataset = TensorDataset(train_images, train_labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = TensorDataset(test_images, test_labels)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==================== 4. 创建模型 ====================

# 判断是否使用GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前使用设备：", device)

model = MLP().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================== 5. 模型训练 ====================

best_val_acc = 0.0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_dataset)


    # ==================== 6. 验证模型 ====================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Validation Accuracy: {val_acc:.4%}"
    )

    # 保存验证集表现最好的模型
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


# ==================== 7. 加载最优模型 ====================

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)


# ==================== 8. 测试集评估 ====================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total

print("\n========== 训练完成 ==========")
print(f"最佳验证集准确率：{best_val_acc:.4%}")
print(f"测试集准确率：{test_acc:.4%}")
print(f"模型权重已保存至：{MODEL_PATH}")