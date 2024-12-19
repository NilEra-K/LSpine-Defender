"""预测代码"""

# 导入需要的包
import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import timm
from timm.utils import ModelEmaV2
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
import re
import pydicom


# 配置: Config
RD = 'workstation/data/rsna2024_small'
OUTPUT_DIR = 'workstation/competition/one-stage/rsna24-results'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = os.cpu_count()
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

N_FOLDS = 5

MODEL_NAME = "tf_efficientnet_b4.ns_jft_in1k"

BATCH_SIZE = 1

CONDITIONS = [
    'spinal_canal_stenosis', 
    'left_neural_foraminal_narrowing', 
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]

LEVELS = [
    'l1_l2',
    'l2_l3',
    'l3_l4',
    'l4_l5',
    'l5_s1',
]

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def load_data(rd):
    df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
    return df


class RSNA24TestDataset(Dataset):
    def __init__(self, df, study_ids, phase='test', transform=None):
        self.df = df
        self.study_ids = study_ids
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.study_ids)
    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'{RD}/test_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(IMG_SIZE[0], IMG_SIZE[1])
        return img

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        st_id = self.study_ids[idx]        
        
        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass
            
        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2) == 0:
            print(st_id, ': Sagittal T2/STIR, has no images')
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2) + 0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i - 0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j + 10] = img.astype(np.uint8)
                except:
                    print(f'[ERROR] Failed to Load ON {st_id}, Sagittal T2/STIR')
                    pass
            
        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')
            
        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass  

        if self.transform is not None:
            x = self.transform(image=x)['image']
            
        x = x.transpose(2, 0, 1)
        return x, str(st_id)


sys.path.append('workstation/lib/efficient-kan-master/src')
from efficient_kan import KAN

def replace_linear_with_kan(module):
    for name, sub_module in module.named_children():
        # if sub_module is a linear layer (MLP), replace it with KAN
        if isinstance(sub_module, nn.Linear):
            in_features = sub_module.in_features
            out_features = sub_module.out_features
            # hidden_layers = min(in_features, out_features)  # or other way to determine hidden_layers
            hidden_layers = min(in_features, out_features) * 2
            setattr(module, name, KAN([in_features, hidden_layers, out_features]))
        else:
            replace_linear_with_kan(sub_module)

class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained, 
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )
        replace_linear_with_kan(self.model)
    
    def forward(self, x):
        y = self.model(x)
        return y


def load_models():
    models = []
    CKPT_PATHS = glob.glob('workstation/competition/one-stage/rsna24-results/best_wll_model_fold-*.pt')
    CKPT_PATHS = sorted(CKPT_PATHS)
    for i, cp in enumerate(CKPT_PATHS):
        print(f'[INFO] LOADING {cp}...')
        model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(cp))
        model.eval()
        model.half()
        model.to(device)
        models.append(model)
    return models


# 定义 autocast
# autocast 是自动混合精度功能的上下文管理器
# 用于临时改变默认的数据类型, 以便在执行特定的操作时使用更低精度的数据类型 (通常是半精度, 即 torch.half)
# 从而减少计算资源的使用, 加快训练速度, 并可能减少内存消耗
def get_autocast():
    # 根据版本不同选择合适的代码
    # autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16) # if your gpu is newer Ampere, you can use this, lesser appearance of nan than half
    # autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)     # you can use with T4 gpu. or newer
    autocast = torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.half)
    return autocast


# 定义 scaler
# scaler 是 AMP 自动混合精度功能的梯度缩放器
# 防止梯度下溢, 特别是在使用半精度 (float16) 进行训练时
def get_scaler():
    # scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP, init_scale=4096)
    return scaler

# AMP(Automatic Mixed Precision) 自动混合精度
def get_amp_config():
    return get_autocast(), get_scaler()

def inference_model():
    print('[IMPORTANT] START inference_model() ...')
    autocast, _ = get_amp_config()
    df = load_data(RD)
    study_ids = list(df['study_id'].unique())
    transforms_test = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])
    
    test_ds = RSNA24TestDataset(df, study_ids, transform=transforms_test)
    test_dl = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    models = load_models()
    y_preds = []
    row_names = []

    with tqdm(test_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, (x, si) in enumerate(pbar):
                x = x.to(device)
                pred_per_study = np.zeros((25, 3))
                
                for cond in CONDITIONS:
                    for level in LEVELS:
                        row_names.append(si[0] + '_' + cond + '_' + level)
                
                with autocast:
                    for m in models:
                        y = m(x)[0]
                        for col in range(N_LABELS):
                            pred = y[col*3:col*3+3]
                            y_pred = pred.float().softmax(0).cpu().numpy()
                            pred_per_study[col] += y_pred / len(models)
                    y_preds.append(pred_per_study)

    y_preds = np.concatenate(y_preds, axis=0)
    # print(y_preds)
    return row_names, y_preds


def output_result(row_names, y_preds):
    sample_sub = pd.read_csv(f'{RD}/sample_submission.csv')
    LABELS = list(sample_sub.columns[1:])
    sub = pd.DataFrame()
    sub['row_id'] = row_names
    sub[LABELS] = y_preds
    # sub.head(25)
    return sub.to_json()
    

def main():
    row_names, y_preds = inference_model()
    print(output_result(row_names=row_names, y_preds=y_preds))
    

if __name__ == "__main__":
    main()