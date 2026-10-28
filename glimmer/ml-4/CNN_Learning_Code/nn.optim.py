import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,64,shuffle=False)
train_data=torchvision.datasets.ImageNet("data_imagenet",split='train',download=True,transform=torchvision.transforms.ToTensor())

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        # self.conv1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,1,padding=2)
        # self.maxpool2=MaxPool2d(2)
        # self.conv3=Conv2d(32,64,5,padding=2)
        # self.maxpool3=MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)
        self.sequential=Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        return self.sequential(x)
Model=model()

optim=torch.optim.SGD(Model.parameters(),lr=0.01)
loss=CrossEntropyLoss()
writer=SummaryWriter("logs_loss")

for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs, targets = data
        outputs = Model(imgs)
        # print(outputs)
        # print(targets)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss
    print(running_loss)
