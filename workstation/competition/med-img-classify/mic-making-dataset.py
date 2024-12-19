# 导入必要的库
import pydicom
import glob
import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import re

# 常量: ROOT_DIR
RD = 'workstation/data/rsna2024_small'

# 将文本转换为整数的辅助函数
def atoi(text):
    return int(text) if text.isdigit() else text

# 自然排序的辅助函数
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# 读取DICOM图像并写入PNG文件的函数
def imread_and_imwirte(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)  # 读取DICOM文件
    image = dicom_data.pixel_array          # 获取像素数组
    # 归一化图像到0-255范围
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)  # 调整图像大小
    assert img.shape == (512, 512)                                      # 确保图像大小为512x512
    cv2.imwrite(dst_path, img)                                          # 将图像写入指定路径

def main():
    # 读取必要的 csv 文件
    dfc = pd.read_csv(f'{RD}/train_label_coordinates.csv')
    df = pd.read_csv(f'{RD}/train_series_descriptions.csv')
    
    # 获取去重后的 series_description
    desc = list(df['series_description'].unique())
    
    # 为每种扫描类型创建输出目录
    scan_types = ['Axial_T2', 'Sagittal_T2_STIR', 'Sagittal_T1']
    for scan_type in scan_types:
        output_dir = f'{RD}/mic-images/{scan_type}'
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个 series_description
    for ds in tqdm(desc, desc='Processing series', unit='series'):
        ds_ = ds.replace('/', '_').replace(' ', '_')
        if ds_ in scan_types:
            pdf = df[df['series_description'] == ds]
            allimgs = []
            
            # 收集当前描述的所有图像
            for _, row in tqdm(pdf.iterrows(), total=len(pdf), desc=f'Collecting images for {ds}', unit='img'):
                pimgs = glob.glob(f'{RD}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
                pimgs = sorted(pimgs, key=natural_keys)
                allimgs.extend(pimgs)
            
            if len(allimgs) == 0:
                print(f'{ds} has no images')
                continue

            # 根据 series_description 处理图片
            scan_type_dir = f'{RD}/mic-images/{ds_}'
            for j, impath in tqdm(enumerate(allimgs), total=len(allimgs), desc=f'Converting images for {ds}', unit='img'):
                dst = f'{scan_type_dir}/{j:05d}.png'
                imread_and_imwirte(impath, dst)


if __name__ == "__main__":
    main()