import os
import torch
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np

# 定义推理函数
def predict_image(image_path, model, class_names, device):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 推理
    with torch.no_grad():
        inputs = image.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    # 获取预测标签
    predicted_label = class_names[preds.item()]

    return predicted_label

# 定义推理函数
def predict_image_with_prob(image_path, model, class_names, device):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(512),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # ImageNet的均值和标准差
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 推理
    with torch.no_grad():  # 不计算梯度
        inputs = image.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 计算概率

    # 获取预测标签和概率
    preds = torch.argmax(probabilities, dim=1)
    predicted_label = class_names[preds.item()]
    probabilities = probabilities[0].cpu().tolist()  # 转换为列表

    # 将标签和概率匹配
    matched_results = list(zip(class_names, probabilities))

    return predicted_label, matched_results


# 定义推理函数
def predict_image_with_prob_imageLoaded(image, model, class_names, device):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(512),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # ImageNet的均值和标准差
    ])

    # 加载图像
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 推理
    with torch.no_grad():  # 不计算梯度
        inputs = image.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 计算概率

    # 获取预测标签和概率
    preds = torch.argmax(probabilities, dim=1)
    predicted_label = class_names[preds.item()]
    probabilities = probabilities[0].cpu().tolist()  # 转换为列表

    # 将标签和概率匹配
    matched_results = list(zip(class_names, probabilities))

    return predicted_label, matched_results


# 加载模型
def load_model(model_path, class_names, device):
    model = models.resnet50(pretrained=False)  # 以非预训练模式加载模型结构
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # 替换最后的全连接层以匹配你的类别数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 预测每个类别的准确度
def predict_accuracy(data_dir, model, class_names, device):
    correct_pred = {cls: 0 for cls in class_names}
    total_images = {cls: 0 for cls in class_names}

    # 获取所有类别目录
    all_cls_dirs = [os.path.join(data_dir, cls) for cls in class_names]

    # 创建一个tqdm进度条
    total_files = sum(len([f for f in os.listdir(d) if f.endswith(".png") or f.endswith(".jpg")]) for d in all_cls_dirs)
    pbar = tqdm(total=total_files, desc="Predicting")

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.endswith(".png") or img_name.endswith(".jpg"):
                image_path = os.path.join(cls_dir, img_name)
                true_label = cls
                predicted_label = predict_image(image_path, model, class_names, device)
                total_images[cls] += 1
                if predicted_label == true_label:
                    correct_pred[cls] += 1
                # 更新进度条
                pbar.update(1)

    # 关闭进度条
    pbar.close()

    accuracy = {cls: correct_pred[cls] / total_images[cls] * 100 for cls in class_names}
    return accuracy


def main():
    class_names = ["Axial_T2", "Sagittal_T1", "Sagittal_T2_STIR"]
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = './mic-results/resnet50_model.pth'
    model = load_model(model_path, class_names, device)

    # 预测准确度
    data_dir = 'E:/RSNA-Dataset/mic-split/test_images'      # 替换数据集路径
    accuracy = predict_accuracy(data_dir, model, class_names, device)

    # 打印每个类别的准确度
    for cls, acc in accuracy.items():
        print(f'Accuracy for {cls}: {acc:.2f}%')
        
if __name__ == "__main__":
    main()