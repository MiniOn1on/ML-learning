import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes=2, pretrained=True):
    """构建 ResNet18 二分类模型，类别顺序为 cat、dog。"""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
