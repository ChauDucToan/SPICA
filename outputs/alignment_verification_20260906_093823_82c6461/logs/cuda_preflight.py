import os
import sys
import torch

print('python', sys.executable, 'torch', torch.__version__, 'build_cuda', torch.version.cuda)
print('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH'))
print('cuda_available', torch.cuda.is_available())
assert torch.cuda.is_available()
print('gpu', torch.cuda.get_device_name(), 'memory_free_total', torch.cuda.mem_get_info())
x = torch.randn(128, 128, device='cuda', requires_grad=True)
y = (x @ x.T).square().mean()
y.backward()
torch.cuda.synchronize()
assert torch.isfinite(y) and torch.isfinite(x.grad).all() and x.grad.norm() > 0
print('FORWARD_BACKWARD_SYNCHRONIZE_PASS', y.item(), x.grad.norm().item())
