import torch
import torchvision

print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchaudio 版本: {torch.audio.__version__ if hasattr(torch, 'audio') else 'N/A'}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"当前 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("未检测到可用的 CUDA 设备，将使用 CPU 运行。")

# 验证简单的张量运算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(5, 3, device=device)
print(f"\n测试张量 (在 {device} 上): \n{x}")