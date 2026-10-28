import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


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

if __name__=='__main__':
    model=Model()
    input=torch.ones((64,3,32,32))
    output=model(input)
    print(output.shape)