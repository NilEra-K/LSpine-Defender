import sys
import torch
import importlib

# 动态导入模块
mic_predict = importlib.import_module("mic-predict")

def predict():
    img_path = "workstation/data/rsna2024_small/mic-split/test_images/Axial_T2/00007.png"

    class_names = ["Axial_T2", "Sagittal_T1", "Sagittal_T2_STIR"]
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = 'workstation/competition/med-img-classify/mic-results/resnet50_model.pth'
    model = mic_predict.load_model(model_path, class_names, device)

    # 预测准确度
    predicted_label, probabilities = mic_predict.predict_image_with_prob(img_path, model=model, class_names=class_names, device=device)
    # print(predicted_label, probabilities)
    return predicted_label, probabilities

def main():
    img_path = "workstation/data/rsna2024_small/mic-split/test_images/Axial_T2/00007.png"

    class_names = ["Axial_T2", "Sagittal_T1", "Sagittal_T2_STIR"]
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = 'workstation/competition/med-img-classify/mic-results/resnet50_model.pth'
    model = mic_predict.load_model(model_path, class_names, device)

    # 预测准确度
    predicted_label, probabilities = mic_predict.predict_image_with_prob(img_path, model=model, class_names=class_names, device=device)
    print(predicted_label, probabilities)
    

if __name__ == "__main__":
    main()