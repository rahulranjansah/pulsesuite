from numba import cuda
print(cuda.is_available())

import sys
print(sys.version)
print(sys.executable)


import sys

print("Python version:")
print(sys.version)
print("Python executable:")
print(sys.executable)
print("-" * 40)

try:
    import torch
except ImportError as e:
    print("PyTorch is not installed.")
    print(e)
    sys.exit(1)

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    # Simple GPU sanity test
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = x * 2
    print("GPU tensor test result:", y)
else:
    print("CUDA is NOT available in PyTorch.")
