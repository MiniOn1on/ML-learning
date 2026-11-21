
import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from model import UNet
from dataset import CrackDataset


# 1. 基本配置

DATA_ROOT = "dataset/DeepCrack"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_img")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "test_lab")

MODEL_PATH = "weights/best_model.pth"
RESULT_DIR = "results"

BATCH_SIZE = 1
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 2. 保存预测可视化结果

def save_visualization(image, gt_mask, pred_mask, save_path):
    """
    保存原图、真实Mask、预测Mask和叠加效果图。
    """

    # 转换为 NumPy 数组
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)

    gt_mask = gt_mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    # 转换为 RGB 图像
    original = Image.fromarray(image).convert("RGB")

    gt = Image.fromarray(
        (gt_mask * 255).astype(np.uint8)
    ).convert("RGB")

    pred = Image.fromarray(
        (pred_mask * 255).astype(np.uint8)
    ).convert("RGB")

    # 制作红色预测区域叠加图
    image_array = np.array(original).copy()

    overlay_color = np.zeros_like(image_array)
    overlay_color[:, :, 0] = 255

    mask = pred_mask.astype(bool)

    image_array[mask] = (
        0.5 * image_array[mask] +
        0.5 * overlay_color[mask]
    ).astype(np.uint8)

    overlay = Image.fromarray(image_array)

    # 创建 2×2 对比图
    width, height = original.size

    comparison = Image.new(
        "RGB",
        (width * 2, height * 2),
        color="white"
    )

    comparison.paste(original, (0, 0))
    comparison.paste(gt, (width, 0))
    comparison.paste(pred, (0, height))
    comparison.paste(overlay, (width, height))

    comparison.save(save_path)


# 3. 加载测试数据

test_dataset = CrackDataset(
    img_dir=TEST_IMG_DIR,
    mask_dir=TEST_MASK_DIR,
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("测试图片数量:", len(test_dataset))


# 4. 加载训练好的模型

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("模型加载成功！")
print("当前设备:", DEVICE)


# 5. 测试与预测

os.makedirs(RESULT_DIR, exist_ok=True)

tp_total = 0
fp_total = 0
fn_total = 0

with torch.no_grad():

    for index, (images, masks) in enumerate(test_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # 模型预测
        outputs = model(images)

        # 转换为概率
        probabilities = torch.sigmoid(outputs)

        # 根据阈值转换为二值 Mask
        preds = (probabilities > THRESHOLD).float()

        # 累计 TP、FP、FN
        tp_total += (preds * masks).sum().item()

        fp_total += (
            preds * (1 - masks)
        ).sum().item()

        fn_total += (
            (1 - preds) * masks
        ).sum().item()

        # 获取当前图片名称
        img_path = test_dataset.images[index]
        filename = img_path.stem

        # 保存预测结果
        save_visualization(
            images[0],
            masks[0],
            preds[0],
            os.path.join(
                RESULT_DIR,
                filename + "_comparison.png"
            )
        )

        # 单独保存预测 Mask
        pred_array = (
            preds[0, 0].cpu().numpy() * 255
        ).astype(np.uint8)

        Image.fromarray(pred_array).save(
            os.path.join(
                RESULT_DIR,
                filename + "_pred.png"
            )
        )

        if (index + 1) % 20 == 0:
            print(
                f"已完成 {index + 1}/{len(test_dataset)} 张图片预测"
            )


# 6. 计算整个测试集指标

smooth = 1e-6

iou = tp_total / (
    tp_total + fp_total + fn_total + smooth
)

dice = (2 * tp_total) / (
    2 * tp_total + fp_total + fn_total + smooth
)

precision = tp_total / (
    tp_total + fp_total + smooth
)

recall = tp_total / (
    tp_total + fn_total + smooth
)


# 7. 输出最终结果

print("\n========== 测试集最终结果 ==========")

print(f"测试图片数量: {len(test_dataset)}")

print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("===================================")

print(f"预测结果已保存到: {RESULT_DIR}")
