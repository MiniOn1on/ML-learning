#  我们设定了一个真实的数据生成公式：y=2x_1-3.4x_2+4.2
# （即真实的权重 w=[2,-3.4],偏置b=4.2
# ），并在其中加入了部分高斯噪声。 你需要编写代码来拟合这些数据，通
# 过模型训练找出这组真实的 
# 和 。  
# 你需要完成以下步骤，并提交两份不同版本的代码：  
# 1. 版本一（PyTorch 自动求导与框架版）： 调用 PyTorch 框架，利用其自动梯度机制或直接使用内
# 置的线性层和优化器完成梯度求导和参数更新。  
# 2. 版本二（NumPy 手动推导版）： 在此代码中，严禁依赖 PyTorch 的自动求导机制。你需要自己
# 推导 MSE 损失函数对 
# 和 的偏导数公式，并使用 NumPy 手动计算梯度和更新参数。（提示：
# NumPy 与 PyTorch 的张量不能直接混用，需要显式进行数据类型转换）。  
# 【具体流程要求】
# 实现线性回归模型的前向预测。  
# 定义均方误差损失函数。  
# 实现梯度的计算（自动或手动）。  
# 使用梯度下降法更新模型参数。  
# 输出训练后的最终模型参数。  
# （可选）： 使用 
# matplotlib.pyplot 库绘制出最后拟合的图像。 
import numpy as np
import torch
import matplotlib.pyplot as plt # 用于最后的绘图（可选）
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), 
dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), 
dtype=torch.float)
# 下方为函数模块提示（供参考，可自行调整结构）
# 前向传播
def predict(x, w, b):
    return x @ torch.tensor(w).T + b
# 损失函数
def mse(pre_y, y):
    return torch.mean((pre_y - y) ** 2)
# 梯度计算
def gradient(pre_y, x, y):
    n = x.shape[0]  # 样本数量
    y=y.reshape(-1,1)#（1000，1）
    pre_y=pre_y.reshape(-1,1)
    # 计算预测值与真实值的差值
    diff = pre_y - y

    # 计算梯度，x（1000，2）
    grad_w = 2 * torch.sum(diff * x, dim=0, keepdim=True) / n
    grad_b = 2 * torch.sum(diff, dim=0, keepdim=True) / n
    return grad_w, grad_b
# 梯度更新
def gradient_update(learning_rate, x, y, pre_y, w, b):
    grad_w, grad_b = gradient(pre_y, x, y)
    w_new = torch.tensor(w) - learning_rate * grad_w
    b_new = torch.tensor(b) - learning_rate * grad_b
    return w_new, b_new
# 训练函数
def train(x, y, w, b, epoch, learning_rate):
    losses = []
    for epoch in range(epoch):
        pre_y = predict(x, w, b)
        loss = mse(pre_y, y)
        w, b = gradient_update(learning_rate, x, y, pre_y, w, b)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epoch}], Loss: {loss.item():.6f}')
    return w, b
if __name__ == "__main__":
    w = [0.0, 0.0]
    b = 3.0
    ans_w, ans_b = train(features, labels, w, b, 100, 0.1)
    print(ans_w,ans_b)
# (可选) 绘图部分代码...


