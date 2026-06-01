import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"GPU count:       {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0:           {torch.cuda.get_device_name(0)}")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Tensor on GPU:   {x}")
