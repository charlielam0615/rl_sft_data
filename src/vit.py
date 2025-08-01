import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import timm
import os
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main():
    # 配置参数
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    IMAGENET_PATH = "/data/datasets/imagenet"  # 用户需要修改为实际的ImageNet数据集路径
    MODEL_NAME = "vit_base_patch16_224"  # ViT基础模型

    # 检查设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")

    # 定义数据预处理变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载ImageNet测试集
    try:
        test_dataset = ImageNet(
            root=IMAGENET_PATH,
            split='val',
            transform=transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"成功加载测试集，共 {len(test_dataset)} 张图片")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保ImageNet数据集路径正确，并且包含val文件夹和对应的标注文件")
        return

    # 加载预训练ViT模型
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=1000
    ).to(device)
    model.eval()
    print(f"成功加载预训练模型: {MODEL_NAME}")

    # 初始化准确率计算变量
    top1_correct = 0
    top5_correct = 0
    total = 0

    # 开始评估
    print("开始评估...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算Top-1和Top-5准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()

            # Top-5计算
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            top5_correct += sum([labels[j] in top5_pred[j] for j in range(labels.size(0))])

            # 打印进度
            if (i + 1) % 50 == 0:
                print(f"批次 [{i + 1}/{len(test_loader)}], 当前Top-1准确率: {100 * top1_correct / total:.2f}%")

    # 计算最终准确率
    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total

    # 输出结果
    result = f"ViT模型在ImageNet-1k测试集上的准确率:\n"
    result += f"Top-1准确率: {top1_acc:.2f}%\n"
    result += f"Top-5准确率: {top5_acc:.2f}%"
    print(result)

    # 保存结果到文件
    with open("vit_imagenet_accuracy_result.md", "w", encoding="utf-8") as f:
        f.write("# ViT模型ImageNet-1k准确率评估结果\n\n")
        f.write(f"## 模型信息\n- 模型名称: {MODEL_NAME}\n- 预训练权重: 是\n\n")
        f.write(f"## 评估结果\n- Top-1准确率: **{top1_acc:.2f}%**\n- Top-5准确率: **{top5_acc:.2f}%**\n\n")
        f.write(f"## 评估参数\n- 批处理大小: {BATCH_SIZE}\n- 使用设备: {device.type}\n- 测试集样本数: {total}\n")

    print("评估结果已保存到 vit_imagenet_accuracy_result.md")


if __name__ == "__main__":
    main()