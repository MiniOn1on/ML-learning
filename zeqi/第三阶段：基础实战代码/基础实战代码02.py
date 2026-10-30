# 我们设定了一个真实的数据生成公式：
# ，偏置
# （即真实的权重
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
# 【初始代码模板】 请基于以下代码模板进行补充和完善：
import numpy as np
import torch
import torch.nn as nn
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
labels=labels.unsqueeze(1)
# 下方为函数模块提示（供参考，可自行调整结构）
class Linear2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear1=nn.Linear(2,1)
    def forward(self,x):
        return self.linear1(x)
model=Linear2D()

# 损失函数
loss_fn=nn.MSELoss()
#学习率
learning_rate=0.01
# 梯度计算
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1000
loss_history = []

# 训练函数
def train(x, y):
    for epoch in range(epochs):
        y_pred=model(x)
        loss=loss_fn(y_pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
train(features,labels)
w = model.linear1.weight
b = model.linear1.bias
if __name__ == "__main__":

    print(w,b)
# (可选) 绘图部分代码...