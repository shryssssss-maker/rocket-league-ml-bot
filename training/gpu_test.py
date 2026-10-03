"""Fail clearly if CUDA cannot execute actual PyTorch work."""
import sys
import torch


def main():
    print('Python:', sys.version.split()[0])
    print('torch.__version__:', torch.__version__)
    print('Bundled CUDA runtime:', torch.version.cuda)
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: stop before PPO; inspect driver and PyTorch build')
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    torch.manual_seed(42)
    a = torch.randn(64, 64, device='cuda')
    b = torch.randn(64, 64, device='cuda')
    result = a @ b
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), a.cpu() @ b.cpu(), rtol=1e-4, atol=1e-4)
    assert '4050' in torch.cuda.get_device_name(0)
    print('PASS: RTX 4050 CUDA matrix multiplication agrees with CPU')


if __name__ == '__main__':
    main()
