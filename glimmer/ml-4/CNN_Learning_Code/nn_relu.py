import  torch
import torchvision
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

dataset=torchvision.datasets.CIFAR10("data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,shuffle=False)

class nn(nn.Module):
    def __init__(self):
        super(nn,self).__init__()
        self.relu1=ReLU(inplace=False)# inplace=True原地修改，False需要再赋值
        self.sigmoid1=Sigmoid()
    def forward(self,x):
        output=self.sigmoid1(x)
        return output

NN=nn()
writer=SummaryWriter("logs_relu")
step=0
for data in dataloader:
    imgs,target=data
    writer.add_images("input",imgs,step)
    output=NN(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()
