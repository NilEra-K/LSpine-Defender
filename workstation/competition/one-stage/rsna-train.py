# RSNA2024 LSDC Training Baseline
# 训练模型代码: 需要先执行 rsna-making-dataset.py

# 导入需要的库
import os
import gc
import sys
from PIL import Image
import cv2
import math
import random
import numpy as np
import pandas as pd
from glob import glob
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
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
from sklearn.metrics import log_loss

# 配置项 Config
NOT_DEBUG = False                                           # True -> 正常运行, False -> 调试模式

OUTPUT_DIR = 'rsna24-results'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # 选择合适的运算设备
N_WORKERS = os.cpu_count()                                  # 选择合适的工作线程数
USE_AMP = True                                              # 如果使用T4或更新的Ampere可以更改为True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75

N_FOLDS = 5 if NOT_DEBUG else 2
EPOCHS = 20 if NOT_DEBUG else 2
MODEL_NAME = "tf_efficientnet_b3.ns_jft_in1k" if NOT_DEBUG else "tf_efficientnet_b0.ns_jft_in1k"

GRAD_ACC = 2
TGT_BATCH_SIZE = 2
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True

# 常量: ROOT_DIR
RD = 'workstation/data/rsna2024_small'

CONDITIONS = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)                        # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

set_random_seed(SEED)

# 打开数据框
def load_data(rd):
    df = pd.read_csv(f'{rd}/train.csv')
    df = df.fillna(-100)                    # 将NaN替换为-100
    label2id = { 'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2 }
    df = df.replace(label2id)
    return df

class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((512, 512, IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                p = f'{RD}/cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T1')
                pass
            
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{RD}/cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass
            
        # Axial T2
        axt2 = glob(f'{RD}/cvt_png/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)
    
        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2)+0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass  
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label

# 定义数据增强
def get_transforms():
    transforms_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=AUG_PROB),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=AUG_PROB),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
        A.Normalize(mean=0.5, std=0.5)
    ])

    transforms_val = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])

    if not NOT_DEBUG or not AUG:
        transforms_train = transforms_val
        
    return transforms_train, transforms_val


def try_dataloader():
    df = load_data(RD)
    transforms_train, transforms_val = get_transforms()
    tmp_ds = RSNA24Dataset(df, phase='train', transform=transforms_train)
    tmp_dl = DataLoader(
        tmp_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    for i, (x, t) in enumerate(tmp_dl):
        if i==5:break
        print('x stat:', x.shape, x.min(), x.max(),x.mean(), x.std())
        print(t, t.shape)
        y = x.numpy().transpose(0,2,3,1)[0,...,:3]
        y = (y + 1) / 2
        plt.imshow(y)
        plt.show()
        print('y stat:', y.shape, y.min(), y.max(),y.mean(), y.std())
        print()
    plt.close()
    del tmp_ds, tmp_dl


# 定义模型
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
    
    def forward(self, x):
        y = self.model(x)
        return y


def try_model():
    m = RSNA24Model(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False)
    i = torch.randn(2, IN_CHANS, 512, 512)
    out = m(i)
    for o in out:
        print(o.shape, o.min(), o.max())
    del m, i, out


# 定义 autocast
# autocast 是自动混合精度功能的上下文管理器
# 用于临时改变默认的数据类型, 以便在执行特定的操作时使用更低精度的数据类型 (通常是半精度, 即 torch.half)
# 从而减少计算资源的使用, 加快训练速度, 并可能减少内存消耗
def get_autocast():
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

# 训练循环
def train_model(df, transforms_train, transforms_val):
    print('[IMPORTANT] START train_model() ...')
    autocast, scaler = get_amp_config()

    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print(f'[INFO] Start Fold-{fold}')
        print(f'[INFO] Len trn_idx: {len(trn_idx)}, len val_idx: {len(val_idx)}')
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_ds = RSNA24Dataset(df_train, phase='train', transform=transforms_train)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )

        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE*2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )
        print("[INFO] len(train_dl): ", len(train_dl), " len(valid_dl): ", len(valid_dl))

        model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=True)
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

        warmup_steps = EPOCHS/10 * len(train_dl) // GRAD_ACC
        num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_total_steps,
            num_cycles=num_cycles
        )

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        best_loss = 1.2
        best_wll = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS+1):
            print(f'[INFO] Start Epoch {epoch}...')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):  
                    x = x.to(device)
                    t = t.to(device)
                    
                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col]
                            loss = loss + criterion(pred, gt) / N_LABELS
                            
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC
        
                    if not math.isfinite(loss):
                        print(f"[INFO] Loss IS {loss}, Stopping Training")
                        sys.exit(1)
        
                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item()*GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)
                    
                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()                    
        
            train_loss = total_loss/len(train_dl)
            print(f'[INFO] train_loss: {train_loss:.6f}')

            total_loss = 0
            y_preds = []
            labels = []
            
            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):
                        
                        x = x.to(device)
                        t = t.to(device)
                            
                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model(x)
                            for col in range(N_LABELS):
                                pred = y[:,col*3:col*3+3]
                                gt = t[:,col]
    
                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())
                            
                            total_loss += loss.item()   
        
            val_loss = total_loss/len(valid_dl)
            
            y_preds = torch.cat(y_preds, dim=0)
            labels = torch.cat(labels)
            val_wll = criterion2(y_preds, labels)
            
            print(f'[INFO] val_loss: {val_loss:.6f}, val_wll: {val_wll:.6f}')

            if val_loss < best_loss or val_wll < best_wll:
                es_step = 0
                if device!='cuda:0':
                    model.to('cuda:0')                
                    
                if val_loss < best_loss:
                    print(f'[INFO] Epoch:{epoch}, Best Loss Updated From {best_loss:.6f} TO {val_loss:.6f}')
                    best_loss = val_loss
                    
                if val_wll < best_wll:
                    print(f'[INFO] Epoch:{epoch}, Best wll_metric Updated From {best_wll:.6f} TO {val_wll:.6f}')
                    best_wll = val_wll
                    fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                    torch.save(model.state_dict(), fname)
                
                if device!='cuda:0':
                    model.to(device)
                
            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('[IMPORTANT] EARLY STOPPING...')
                    break  
                                

def calculation_cv():
    print('[IMPORTANT] START calculation_cv()...')
    autocast, scaler = get_amp_config()

    cv = 0
    y_preds = []
    labels = []
    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion2 = nn.CrossEntropyLoss(weight=weights)

    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df = load_data(RD)
    transforms_train, transforms_val = get_transforms()
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print('#'*30)
        print(f'start fold{fold}')
        print('#'*30)
        df_valid = df.iloc[val_idx]
        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )

        model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=False)
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
        model.load_state_dict(torch.load(fname))
        model.to(device)   
        
        model.eval()
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):           
                    x = x.to(device)
                    t = t.to(device)
                        
                    with autocast:
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col] 
                            y_pred = pred.float()
                            y_preds.append(y_pred.cpu())
                            labels.append(gt.cpu())

    y_preds = torch.cat(y_preds)
    labels = torch.cat(labels)
    cv = criterion2(y_preds, labels)
    print('[INFO] CV Score:', cv.item())
    
    y_pred_np = y_preds.softmax(1).numpy()
    labels_np = labels.numpy()
    y_pred_nan = np.zeros((y_preds.shape[0], 1))
    y_pred2 = np.concatenate([y_pred_nan, y_pred_np],axis=1)
    weights = []
    for l in labels:
        if l==0: weights.append(1)
        elif l==1: weights.append(2)
        elif l==2: weights.append(4)
        else: weights.append(0)
    cv2 = log_loss(labels, y_pred2, normalize=True, sample_weight=weights)
    print('[INFO] Calculation Competition Metrics -> CV Score: ', cv2)
    np.save(f'{OUTPUT_DIR}/labels.npy', labels_np)
    np.save(f'{OUTPUT_DIR}/final_oof.npy', y_pred2)
    print(f'[INFO] labels.npy & final_oof.npy Has Been Saved IN {OUTPUT_DIR}')
    
    random_pred = np.ones((y_preds.shape[0], 3)) / 3.0
    y_pred3 = np.concatenate([y_pred_nan, random_pred],axis=1)
    cv3 = log_loss(labels, y_pred3, normalize=True, sample_weight=weights)
    print('[INFO] Random Score:', cv3)


# 主函数
def main():
    # 常量: ROOT_DIR
    # RD = 'workstation/data/rsna2024_small'
    df = load_data(RD)
    transforms_train, transforms_val = get_transforms()
    try_dataloader()
    try_model()
    train_model(df, transforms_train, transforms_val)
    calculation_cv()

if __name__ == "__main__":
    main()