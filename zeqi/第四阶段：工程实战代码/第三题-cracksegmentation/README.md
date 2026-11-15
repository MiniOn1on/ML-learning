# 基于 U-Net 的道路裂缝图像分割

本项目使用 **PyTorch** 从零搭建 U-Net，实现道路裂缝的像素级分割。模型输入道路图像，输出与原图尺寸一致的二值裂缝掩码（Mask），并在测试阶段计算分割指标、保存预测结果和可视化对比图。

## 1. 项目功能

- 使用 DeepCrack 数据集进行训练和测试。
- 使用 PyTorch 实现 U-Net 网络结构。
- 对输入图像进行预处理，并对训练数据进行随机翻转、旋转和亮度/对比度增强。
- 使用 BCE Loss 与 Dice Loss 的加权组合作为训练损失。
- 使用 IoU、Dice、Precision 和 Recall 评价分割效果。
- 保存验证集 Dice 最优的模型权重。
- 对测试集进行预测，并将预测裂缝区域以半透明红色遮罩叠加到原图上，保存对比结果。

## 2. 数据集

本项目使用 [DeepCrack](https://github.com/yhlleo/DeepCrack) 数据集。

将数据集整理为以下目录结构：

```text
dataset/
└── DeepCrack/
    ├── train_img/   # 训练图像（300 张）
    ├── train_lab/   # 训练标签/真实 Mask（300 张）
    ├── test_img/    # 测试图像（237 张）
    ├── test_lab/    # 测试标签/真实 Mask（237 张）
    └── README.md    # 数据集说明
```

图像与标签需要按文件名一一对应。标签为黑白二值图像，程序会将其转换为 0/1：背景为 0，裂缝为 1。

训练时，从官方 300 张训练图像中划分出训练集和验证集：
- 训练集：240 张
- 验证集：60 张
- 测试集：使用官方提供的 237 张测试图像



## 3. 项目目录

```text
CrackSegmentation/
├── dataset/
│   └── DeepCrack/
│       ├── train_img/
│       ├── train_lab/
│       ├── test_img/
│       └── test_lab/
├── weights/
│   └── best_model.pth       # 验证集 Dice 最优的模型权重
├── results/                 # 测试集预测结果和可视化图片
├── dataset.py                # 数据集读取、预处理与数据增强
├── model.py                  # U-Net 网络结构
├── train.py                  # 模型训练与验证
├── predict.py                # 测试集预测、指标计算与结果可视化
├── requirements.txt          # Python 依赖
└── README.md
```


## 4. 训练模型

确认 `dataset/DeepCrack/` 下的数据目录和文件名配对正确后，在项目根目录运行：

```bash
python train.py
```

训练脚本会自动选择可用的计算设备（CUDA 或 CPU），并执行训练和验证。训练完成后，验证集 Dice 最优的模型权重会保存到：

```text
weights/best_model.pth
```

训练轮数、Batch Size、学习率和数据集路径等参数可在 `train.py` 文件顶部修改。

## 5. 测试与预测

确保已经完成模型训练，并且存在 `weights/best_model.pth`，然后运行：

```bash
python predict.py
```

程序会加载最佳模型，对官方测试集进行预测，计算整体像素级指标，并将结果保存到 `results/` 目录。

每张测试图片会生成：
- `*_pred.png`：模型预测得到的二值裂缝掩码。
- `*_comparison.png`：四图对比结果，包含原始图像、真实 Mask、预测 Mask，以及将预测裂缝区域以半透明红色叠加到原图后的可视化结果。

## 6. 评价指标

本项目使用以下指标评价裂缝分割效果：

| 指标 | 含义 |
|---|---|
| IoU | 预测裂缝区域与真实裂缝区域的交并比 |
| Dice | 预测区域与真实区域的重叠程度 |
| Precision | 被预测为裂缝的像素中，实际为裂缝的比例 |
| Recall | 真实裂缝像素中，被模型正确识别的比例 |

其中，TP 表示正确预测为裂缝的像素数，FP 表示错误预测为裂缝的背景像素数，FN 表示被错误预测为背景的裂缝像素数。

## 7. 当前实验结果

以下为本次实验记录的结果，供复现和参考：

### 验证集最佳结果

| 指标 | 数值 |
|---|---:|
| Dice | 0.8553 |
| IoU | 0.7489 |
| Precision | 0.8567 |
| Recall | 0.8603 |

### 测试集结果

| 指标 | 数值 |
|---|---:|
| IoU | 0.7053 |
| Dice | 0.8272 |
| Precision | 0.9106 |
| Recall | 0.7577 |


## 8. 参考资料(dataset来源)

1. Liu, Y. et al. *DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation*. Neurocomputing, 338, 139–153, 2019. DOI: [10.1016/j.neucom.2019.01.036](https://doi.org/10.1016/j.neucom.2019.01.036)
2. DeepCrack 数据集及项目主页：[https://github.com/yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack)
