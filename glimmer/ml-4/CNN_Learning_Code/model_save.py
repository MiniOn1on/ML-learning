import torch
import torchvision.models

vgg16=torchvision.models.vgg16(pretrained=False)
#保存方式1
torch.save(vgg16,"vgg16_method1.pth")# 保存模型+参数
#保存方式2
torch.save(vgg16.state_dict(),"vgg16_method2.pth")# 保存模型参数（官方推荐，占用更小）

# 保存方式1只有在模型完全相同时才可以使用