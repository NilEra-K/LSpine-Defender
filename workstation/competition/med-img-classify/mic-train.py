import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

N_WORKERS = 0
class_names = ["Axial_T2", "Sagittal_T1", "Sagittal_T2_STIR"]

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集的根目录
        :param transform: 应用于图像的转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历目录，收集图像路径和标签
        for label_dir in os.listdir(root_dir):
            label_dir_path = os.path.join(root_dir, label_dir)
            for image_name in os.listdir(label_dir_path):
                image_path = os.path.join(label_dir_path, image_name)
                self.images.append(image_path)
                self.labels.append(label_dir)

    def __len__(self):
        """
        返回数据集中的图像数量
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取单个图像及其标签
        """
        image_path = self.images[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image, label
    
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据路径
data_dir = 'workstation/data/rsna2024_small/mic-split'

# 创建数据集
train_dataset = CustomDataset(root_dir=os.path.join(data_dir, 'train_images'), transform=data_transforms['train'])
valid_dataset = CustomDataset(root_dir=os.path.join(data_dir, 'valid_images'), transform=data_transforms['valid'])
test_dataset = CustomDataset(root_dir=os.path.join(data_dir, 'test_images'), transform=data_transforms['test'])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=N_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=N_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=N_WORKERS)

# 从训练数据加载器中获取一批数据
batch_images, batch_labels = next(iter(train_loader))

# 显示一批图像
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 反标准化
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一点时间，以便更新绘图

# 选择几个图像进行显示
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
for i, (img, label) in enumerate(zip(batch_images, batch_labels)):
    if i == 6:
        break
    axes[i].imshow(img.permute(1, 2, 0))  # 调整通道顺序以适应matplotlib
    axes[i].set_title(class_names[label])
    axes[i].axis('off')
plt.show()