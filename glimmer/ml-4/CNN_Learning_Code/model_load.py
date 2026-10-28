import torch
import torchvision.models
from torchvision.models import vgg16

# 方式1->保存方式1，加载模型

mode=torch.load("vgg16_method1.pth",weights_only=False)
print(mode)

# 方式2->加载模型
param=torch.load("vgg16_method2.pth")
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(param)
print(vgg16)

