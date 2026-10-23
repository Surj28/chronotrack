"""Import this before cellpose/torch-using code to keep the NVIDIA driver untouched.

Even CellposeModel(gpu=False) and CUDA_VISIBLE_DEVICES="" still open
/dev/nvidiactl when PyTorch probes for CUDA; with a faulty GPU (B34 machine,
2026-10-14: Xid 119 and machine-check errors under load) that probe can hang
the process in the kernel. Reporting CUDA as unavailable stops the probe:
checked with strace, zero /dev/nvidia* opens.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
