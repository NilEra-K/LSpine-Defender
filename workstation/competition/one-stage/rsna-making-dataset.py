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
    image = dicom_data.pixel_array  # 获取像素数组
    # 归一化图像到0-255范围
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)  # 调整图像大小
    assert img.shape == (512, 512)  # 确保图像大小为512x512
    cv2.imwrite(dst_path, img)  # 将图像写入指定路径

def main():
    # 读取必要的 csv 文件
    dfc = pd.read_csv(f'{RD}/train_label_coordinates.csv')
    df = pd.read_csv(f'{RD}/train_series_descriptions.csv')
    
    # 获取去重后的 study_id 和 series_description
    st_ids = df['study_id'].unique()
    desc = list(df['series_description'].unique())
    
    # 处理每个 study_id
    for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = df[df['study_id'] == si]  # 获取当前study_id的相关数据
        for ds in desc:
            ds_ = ds.replace('/', '_')
            pdf_ = pdf[pdf['series_description'] == ds]
            os.makedirs(f'{RD}/cvt_png/{si}/{ds_}', exist_ok=True)
            allimgs = []
            
            # 收集当前study和描述的所有图像
            for _, row in pdf_.iterrows():
                pimgs = glob.glob(f'{RD}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')  # 获取DICOM图像路径
                pimgs = sorted(pimgs, key=natural_keys)  # 自然排序图像路径
                allimgs.extend(pimgs)  # 添加到所有图像列表
            
            if len(allimgs) == 0:  # 如果没有图像
                print(si, ds, 'has no images')  # 输出提示信息
                continue  # 跳过当前循环

            # 根据 series_description 处理图片
            if ds == 'Axial T2':
                for j, impath in enumerate(allimgs):
                    dst = f'{RD}/cvt_png/{si}/{ds}/{j:03d}.png'
                    imread_and_imwirte(impath, dst)
                    
            elif ds in ['Sagittal T2/STIR', 'Sagittal T1']:
                step = len(allimgs) / 10.0
                st = len(allimgs)/2.0 - 4.0*step
                end = len(allimgs)+0.0001
                
                for j, i in enumerate(np.arange(st, end, step)):
                    dst = f'{RD}/cvt_png/{si}/{ds_}/{j:03d}.png'
                    ind2 = max(0, int((i-0.5001).round()))
                    imread_and_imwirte(allimgs[ind2], dst)
                
                assert len(glob.glob(f'{RD}/cvt_png/{si}/{ds_}/*.png')) == 10  # 确保输出图像数量为10

if __name__ == "__main__":
    main()