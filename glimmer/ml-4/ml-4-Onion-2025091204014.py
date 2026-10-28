import numpy as np
from torch import nn
from torch.nn import Sequential


class My_Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 统一转为元组，支持方形/矩形卷积核
        self.kh, self.kw = kernel_size,kernel_size
        self.stride = stride
        self.padding = padding

        # ========== 权重初始化 ==========
        scale = np.sqrt(2.0 / (self.in_channels * self.kh * self.kw))
        self.weight = np.random.randn(out_channels, in_channels, self.kh, self.kw) * scale
        # bias: 每个输出通道对应1个偏置，shape [out_channels]
        self.bias = np.zeros((out_channels,))

    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        # 计算输出特征图高、宽
        H_out = (H_in - self.kh + 2 * self.padding) // self.stride + 1
        W_out = (W_in - self.kw + 2 * self.padding) // self.stride + 1

        # 对输入四周补0
        x_pad = np.pad(
            x,
            pad_width=((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=0
        )

        # 创建输出数组，初始全0
        out = np.zeros((N, self.out_channels, H_out, W_out))

        # 四重循环滑动卷积（互相关运算，pytorch Conv2d默认就是互相关）
        for n in range(N):  # 遍历batch中每一张图片
            for oc in range(self.out_channels): # 遍历每一个卷积核（输出通道）
                for i in range(H_out):
                    for j in range(W_out):
                        # 计算当前窗口在pad后图像上的起始坐标
                        h_start = i * self.stride
                        w_start = j * self.stride
                        # 取出局部窗口 patch: [C_in, kh, kw]
                        patch = x_pad[n, :, h_start:h_start+self.kh, w_start:w_start+self.kw]
                        # 点乘求和，再加bias
                        val = np.sum(patch * self.weight[oc])
                        val += self.bias[oc]
                        out[n, oc, i, j] = val
        return out


class My_MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kh, self.kw = kernel_size, kernel_size
        # stride默认等于窗口大小
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        N, C, H_in, W_in = x.shape
        # 计算输出高宽
        H_out = (H_in - self.kh) // self.stride + 1
        W_out = (W_in - self.kw) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        hs = i * self.stride
                        ws = j * self.stride
                        patch = x[n, c, hs:hs+self.kh, ws:ws+self.kw]
                        out[n, c, i, j] = np.max(patch)
        return out


class My_AvgPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kh, self.kw = kernel_size, kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        N, C, H_in, W_in = x.shape
        H_out = (H_in - self.kh) // self.stride + 1
        W_out = (W_in - self.kw) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        hs = i * self.stride
                        ws = j * self.stride
                        patch = x[n, c, hs:hs+self.kh, ws:ws+self.kw]
                        out[n, c, i, j] = np.mean(patch)
        return out

class Model(nn.Module):
    '''输入：3 × 32 × 32

    卷积层：Conv2d(3, 16, kernel_size=3, padding=1)
    激活层：ReLU
    池化层：MaxPool2d(kernel_size=2, stride=2)

    卷积层：Conv2d(16, 32, kernel_size=3, padding=1)
    激活层：ReLU
    池化层：MaxPool2d(kernel_size=2, stride=2)

    全局平均池化：b
    全连接层：Linear(32, 10)'''
    def __init__(self):
        super(Model,self).__init__()
        self.seq=Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出 [N,32,1,1]
            nn.Flatten(),  # [N,32]
            nn.Linear(32,10)
        )
    def forward(self,x):
        return self.seq(x)
