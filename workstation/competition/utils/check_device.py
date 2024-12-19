import torch

def check_device():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
def main():
    check_device()

if __name__ == "__main__":
    main()