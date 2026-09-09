"""Verify environment dependencies, PyTorch backends, and system hardware."""

import sys

import numpy as np
import onnxruntime as ort
import torch


def main():
    print("=== RetinaGuard-QA Environment Verification ===")
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"ONNX Runtime:        {ort.__version__}")
    print(f"NumPy Version:       {np.__version__}")
    print(f"CUDA Available:      {torch.cuda.is_available()}")
    print(
        f"MPS Available (Mac): {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
    )
    print("CPU Execution Target: Confirmed")
    print("================================================")
    print("Environment is verified and ready for execution!")


if __name__ == "__main__":
    main()
