import torch
from torch import nn


class MLP(nn.Module):
    """多层感知机：输入 784 维像素，输出 10 个数字类别。"""

    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 兼容输入形状为 [batch, 784] 或 [batch, 1, 28, 28]
        x = x.view(x.size(0), -1)
        return self.network(x)
