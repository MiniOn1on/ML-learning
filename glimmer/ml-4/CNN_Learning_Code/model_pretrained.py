import torchvision
from torch import nn

# train_data=torchvision.datasets.ImageNet("data_imagenet",split='train',download=True,transform=torchvision.transforms.ToTensor())
vgg16_false=torchvision.models.vgg16(pretrained=False)
vgg16_true=torchvision.models.vgg16(pretrained=True)#已训练完成


vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))# 加一个1000到10输出的线形层
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)