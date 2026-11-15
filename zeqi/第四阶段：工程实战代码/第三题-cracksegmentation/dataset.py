import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance


class CrackDataset(Dataset):

    def __init__(self, img_dir, mask_dir,
                 img_size=(544, 384), augment=False):

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        self.img_size = img_size
        self.augment = augment

        # 获取所有图片
        self.images = sorted(
            [p for p in self.img_dir.iterdir()
             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )

        # 建立 Mask 文件名映射
        self.mask_map = {
            p.stem: p for p in self.mask_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        }

        # 检查图片和 Mask 是否匹配
        for img_path in self.images:
            if img_path.stem not in self.mask_map:
                raise FileNotFoundError(
                    f"找不到对应的Mask: {img_path.name}"
                )

        print(f"成功加载 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # 获取图片路径
        img_path = self.images[index]

        # 根据文件名获取对应Mask
        mask_path = self.mask_map[img_path.stem]

        # 读取图片
        image = Image.open(img_path).convert("RGB")

        # 读取Mask
        mask = Image.open(mask_path).convert("L")

        # 同步进行随机水平翻转
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 同步进行随机垂直翻转
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 同步进行90度旋转
        if self.augment:
            k = random.randint(0, 3)

            if k > 0:
                image = image.rotate(90 * k)
                mask = mask.rotate(90 * k)

        # 只对原图进行亮度和对比度增强
        if self.augment:
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.8, 1.2)
                )

            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.8, 1.2)
                )

        # 调整图片尺寸
        image = image.resize(
            self.img_size,
            Image.Resampling.BILINEAR
        )

        # 调整Mask尺寸，必须使用最近邻插值
        mask = mask.resize(
            self.img_size,
            Image.Resampling.NEAREST
        )

        # 转换为numpy数组
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)

        # Mask二值化：0和255转换为0和1
        mask = (mask > 127).astype(np.float32)

        # 转换为Tensor
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask