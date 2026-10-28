import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,shuffle=False)


class Linear(nn.Module):
    def __init__(self):
        super(Linear,self).__init__()
        self.linear1=nn.Linear(in_features=196608,out_features=10)

    def forward(self,x):
        return self.linear1(x)

linear=Linear()

for data in dataloader:
    imgs,targets=data
    output=torch.reshape(imgs,(1,1,1,-1))# torch.flatten()
    output=linear(output)
    print(output.shape)