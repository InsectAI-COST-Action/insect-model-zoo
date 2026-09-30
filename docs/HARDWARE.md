# GPU / CPU and hardware check

[← back to the README](../README.md)

- `--device auto` (the default) uses an **NVIDIA GPU** if PyTorch can see one, then an **Apple Silicon GPU** (mps),
  and otherwise the **CPU**. Every model works on CPU, just more slowly.
- Before a model is loaded, the zoo compares its memory needs with your free GPU memory and RAM. If the GPU is too
  small it **falls back to the CPU** and tells you why. If the GPU runs out of memory halfway, the image is retried on
  the CPU. This is checked for the detector and the classifier separately. (It matters most for large models:
  `sam3` + `bioclip-2.5` together use about 6 GB of GPU memory.)
- `python main.py --check` shows what it found and flags common problems, e.g. *"NVIDIA GPU found but the CPU-only
  PyTorch is installed"*, or a GPU that is newer than the installed PyTorch supports.

```text
OS:       Windows 11
RAM:      36.7 GB free of 63.4 GB
PyTorch:  2.14.0+cu130 (CUDA 13.0 build)
GPU 0:    NVIDIA GeForce RTX 5090 Laptop GPU, 22.6 GB free of 23.9 GB (sm_120)
Default device: cuda:0
Weights:  C:\code\insect-model-zoo\weights (812.4 GB free)

Models on cuda:0:
  insectdct-v8-m       OK | downloaded
  mothbot-mbd-1-1      OK | not downloaded (21 MB)
  sam3                 OK | not downloaded (3.2 GB) | GATED: no Hugging Face token, see docs/GATED_MODELS.md
  ...
```

## Getting the GPU to work

| System | What to do |
|---|---|
| **Windows + NVIDIA** | the normal install gives the CPU version of PyTorch: see [Optional: NVIDIA GPU on Windows](../README.md#optional-nvidia-gpu-on-windows) |
| **Linux + NVIDIA** | nothing: the normal install already includes GPU support (older driver: the same swap with `cu126`) |
| **Mac (Apple Silicon)** | nothing: used automatically (`mps`) |
| **No GPU** | nothing: runs on CPU |

`python main.py --check` shows whether the GPU is used, and tells you when an NVIDIA card is there but PyTorch cannot
use it. Other versions: [pytorch.org](https://pytorch.org/get-started/locally/).
