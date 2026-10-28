import time
import torch
import torchvision.datasets
from sympy.tensor.array.arrayop import Flatten
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, CrossEntropyLoss,Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 训练设备
if torch.cuda.is_available():
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")
print(f"训练设备为{device}")

train_data=torchvision.datasets.CIFAR10("data",train=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
train_data.size=len(train_data)
test_data.size=len(test_data)
# print(test_data.size)
# print(train_data.size)

train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.seq=Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4,64),
            Linear(64,10)
        )
    def forward(self,x):
        return self.seq(x)
model=Model()
model=model.to(device)
# loss
loss_fn=CrossEntropyLoss()
loss_fn=loss_fn.to(device)
# optim
learing_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=learing_rate)


total_train_step=0
total_test_step=0
epoch=50

writer=SummaryWriter("logs_train")
start_time=time.time()

for i in range(epoch):
    # 训练
    model.train()
    print(f"------第{i+1}轮训练开始------")
    for data in train_dataloader:
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)
        # 优化器调优+反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        writer.add_scalar("train_loss",loss.item(),total_train_step)
        if total_train_step%100==0:
            end_time=time.time()
            print(f"总用时{end_time-start_time}秒")
            print(f"训练次数：{total_train_step},损失值：{loss}")
    # 每轮进行一次测试
    model.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print(f"整体测试集上的loss={total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data.size}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data.size,total_test_step)
    total_test_step+=1
    torch.save((model.state_dict(), f"./train_model/trainmodel_{i+1}.pth"))
    print("模型已保存")
writer.close()

