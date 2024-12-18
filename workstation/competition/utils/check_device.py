import torch

def check_device():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)