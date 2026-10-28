import time

import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, CrossEntropyLoss
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
# +++ 新增：混淆矩阵绘图库
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_data=['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']


device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

train_dataset=torchvision.datasets.CIFAR10("data",train=True,transform=torchvision.transforms.ToTensor())
# print(len(train_dataset))
test_dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
train_dataset,validation_dataset=random_split(train_dataset, [45000, 5000])
# print(len(train_dataset))
# print(len(validation_dataset))
# print(len(test_dataset))

train_dataloader=DataLoader(train_dataset,64,shuffle=True)
validation_dataloader=DataLoader(validation_dataset,64,shuffle=False)
test_dataloader=DataLoader(test_dataset,64,shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.seq=Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        return self.seq(x)
model=Model().to(device)

learning_rate=1e-3
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

loss_fn=CrossEntropyLoss()
loss_fn.to(device)

total_train_step=0
total_validation_step=0
epochs=20

writer=SummaryWriter("logs_train")
start_time=time.time()
best=0
best_accuracy = 0.0
for epoch in range(epochs):
    model.train()
    total_train_accuracy=0
    print(f"------第{epoch+1}轮训练开始------")
    for data in train_dataloader:
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        output=model(imgs)
        train_loss=loss_fn(output,targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss",train_loss,total_train_step)
        total_train_step+=1
        train_accuracy=(output.argmax(1)==targets).sum()
        total_train_accuracy+=train_accuracy
        writer.add_scalar("train_accuracy",train_accuracy/len(train_dataset),total_train_step)
        if total_train_step%100==0:
            end_time=time.time()
            print(f"总用时{end_time-start_time}秒")
            print(f"训练次数：{total_train_step},损失值：{train_loss},训练集上的正确率：{total_train_accuracy/len(train_dataset)}")
    # 每次epoch过一次验证集
    model.eval()

    with torch.no_grad():

        total_validation_loss = 0
        total_validation_accuracy = 0

        for data in validation_dataloader:
            imgs,targets=data
            imgs,targets=imgs.to(device),targets.to(device)
            output=model(imgs)
            loss=loss_fn(output,targets)
            total_validation_loss+=loss.item()
            accuracy=(output.argmax(1)==targets).sum()
            total_validation_accuracy+=accuracy
            writer.add_scalar("validation_loss", total_validation_loss, total_validation_step)
            writer.add_scalar("validation_accuracy", total_validation_accuracy/len(validation_dataset), total_validation_step)
            total_validation_step += 1
        print(f"整体验证集上的loss={total_validation_loss}")
        print(f"整体验证集上的正确率：{total_validation_accuracy / len(validation_dataset)}")
    if total_validation_accuracy / len(validation_dataset) >best_accuracy:
        best=epoch+1
        best_accuracy=total_validation_accuracy / len(validation_dataset)
        torch.save(model.state_dict(), f"./train_model/trainmodel_{epoch+1}.pth")
        print(f" 发现更高准确率！新best acc={best_accuracy:.4f}，已保存最新模型")
writer.close()
# ===================== +++新增：【仅可视化第一层卷积核】=====================
print("\n===== 可视化第一层卷积核 =====")
model.load_state_dict(torch.load(f"./train_model/trainmodel_{best}.pth", map_location=device))
model.eval()

# 获取第一层卷积权重 [out_channels, in_channels, k_h, k_w]
conv1_weight = model.seq[0].weight
# 归一化并拼接成网格图
grid_kernels = make_grid(conv1_weight, normalize=True, scale_each=True, nrow=8)
plt.figure(figsize=(8,4))
plt.imshow(grid_kernels.permute(1,2,0).cpu().numpy())
plt.title("Conv1 Kernels (First Convolution Layer)")
plt.axis('off')
plt.savefig("./img/conv1_kernels.png")
plt.show()
# ===================== 卷积核可视化结束 =====================
# 测试集
model.eval()
with torch.no_grad():
    total_test_accuracy=0
    total_test_loss=0
    # +++ 新增：列表保存全部真实标签、预测标签，用于混淆矩阵
    all_true = []
    all_pred = []

    model.load_state_dict(torch.load(f"./train_model/trainmodel_{best}.pth",map_location=torch.device("cuda")))
    for data in test_dataloader:
        imgs,targets=data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)
        total_test_loss+=loss.item()
        accuracy=(outputs.argmax(1)==targets).sum()
        total_test_accuracy+=accuracy
        # +++ 新增：把标签移到cpu，存入列表
        all_true.extend(targets.cpu().numpy())
        all_pred.extend(outputs.argmax(1).cpu().numpy())

    print(f"整体测试集上的loss={total_test_loss}")
    print(f"整体测试集上的正确率：{total_test_accuracy/len(test_dataset)}")

    # +++ 新增：计算混淆矩阵并绘图
    cm = confusion_matrix(all_true, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("CIFAR10 Test Confusion Matrix")
    plt.tight_layout()
    # +++ 新增：保存图片到本地
    plt.savefig("./img/confusion_matrix_test.png")
    plt.show()