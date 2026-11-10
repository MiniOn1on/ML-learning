from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["cat", "dog"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CatDogDataset(Dataset):
    """
    读取扁平目录中的图片：
      cat.0.jpg、cat.1.jpg、dog.0.jpg、dog.1.jpg
    标签由文件名前缀确定，不使用 ImageFolder。
    """
    def __init__(self, train_dir, transform=None):
        self.train_dir = Path(train_dir)
        self.transform = transform
        self.samples = []

        if not self.train_dir.exists():
            raise FileNotFoundError(f"找不到训练图片目录：{self.train_dir}")

        for path in sorted(self.train_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            prefix = path.name.lower().split(".", 1)[0]
            if prefix == "cat":
                label = 0
            elif prefix == "dog":
                label = 1
            else:
                continue

            self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(
                f"在 {self.train_dir} 中没有找到 cat.* 或 dog.* 图片。"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def stratified_split(samples, val_ratio=0.2, seed=42):
    """按猫/狗类别分别划分，尽量保持训练集和验证集类别比例。"""
    rng = random.Random(seed)
    train_indices, val_indices = [], []

    for class_id in (0, 1):
        indices = [i for i, (_, label) in enumerate(samples) if label == class_id]
        rng.shuffle(indices)

        if len(indices) < 2:
            raise ValueError(
                f"类别 {CLASSES[class_id]} 至少需要 2 张图片才能划分训练/验证集。"
            )

        n_val = max(1, int(len(indices) * val_ratio))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices
