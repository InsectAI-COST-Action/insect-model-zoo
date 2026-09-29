"""
Hardware check: which device to use (NVIDIA GPU -> Apple GPU -> CPU) and whether a model is likely to fit.

The memory numbers in the model cards are rough minimums. They matter little for today's detectors, but big models
(e.g. SAM) will need them: the check warns and falls back to CPU before a model is loaded instead of crashing halfway.
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

GB = 1024 ** 3
TORCH_CUDA_HELP = "https://pytorch.org/get-started/locally/"


@dataclass
class Gpu:
    index: int
    name: str
    total_gb: float
    free_gb: float
    capability: str


@dataclass
class Hardware:
    os: str
    cpu: str
    cpu_cores: int
    ram_total_gb: float
    ram_free_gb: float
    torch_version: str
    torch_cuda: str                       # CUDA version torch was built for, "" for a CPU-only build
    gpus: list = field(default_factory=list)
    mps: bool = False                     # Apple Silicon GPU
    notes: list = field(default_factory=list)   # things the user should fix / know


def _nvidia_smi_gpu():
    """Name of an NVIDIA GPU according to the driver, even if torch cannot use it (CPU-only torch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    except Exception:
        return ""


def probe():
    import psutil
    import torch

    vm = psutil.virtual_memory()
    hw = Hardware(os="%s %s" % (platform.system(), platform.release()),
                  cpu=platform.processor() or platform.machine(), cpu_cores=os.cpu_count() or 1,
                  ram_total_gb=vm.total / GB, ram_free_gb=vm.available / GB,
                  torch_version=torch.__version__, torch_cuda=torch.version.cuda or "")

    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            cap = "sm_%d%d" % (major, minor)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:                    # e.g. a GPU this torch build has no kernels for
                total, free = torch.cuda.get_device_properties(i).total_memory, 0
            hw.gpus.append(Gpu(i, torch.cuda.get_device_name(i), total / GB, free / GB, cap))
            if arch_list and cap not in arch_list and not any(a.startswith("compute_") for a in arch_list):
                hw.notes.append("Your GPU (%s, %s) is newer than this PyTorch build supports (%s). Install a newer "
                                "CUDA build of torch, see README 'GPU'." % (torch.cuda.get_device_name(i), cap,
                                                                            " ".join(arch_list)))
    else:
        smi = _nvidia_smi_gpu()
        if smi and not hw.torch_cuda:
            hw.notes.append("An NVIDIA GPU was found (%s) but the installed PyTorch is the CPU-only build, so it runs "
                            "on CPU. To use the GPU:  pip uninstall -y torch torchvision  then  pip install torch "
                            "torchvision --index-url https://download.pytorch.org/whl/cu130  (older driver or GTX "
                            "10-series and older: cu126; see README 'Optional: NVIDIA GPU on Windows')." % smi)
        elif smi:
            hw.notes.append("An NVIDIA GPU was found (%s) but PyTorch cannot use it. Update the NVIDIA driver or "
                            "install a torch build that matches it (%s)." % (smi, TORCH_CUDA_HELP))

    mps = getattr(torch.backends, "mps", None)
    hw.mps = bool(mps and mps.is_available())
    return hw


def refresh(hw):
    """Update free GPU memory (it drops once a model is loaded)."""
    if not hw.gpus:
        return
    import torch
    for g in hw.gpus:
        try:
            g.free_gb = torch.cuda.mem_get_info(g.index)[0] / GB
        except Exception:
            pass


def _note(hw, msg):
    if msg not in hw.notes:
        hw.notes.append(msg)


def pick_device(requested, hw):
    """'auto' -> cuda:0 / mps / cpu. An explicit device that is not available falls back to CPU with a note."""
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if hw.gpus:
            return "cuda:%d" % max(hw.gpus, key=lambda g: g.free_gb).index
        return "mps" if hw.mps else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        idx = int(requested.split(":")[1]) if ":" in requested else 0
        if any(g.index == idx for g in hw.gpus):
            return requested
        _note(hw, "Device '%s' is not available, using CPU." % requested)
        return "cpu"
    if requested == "mps" and not hw.mps:
        _note(hw, "Apple GPU (mps) is not available, using CPU.")
        return "cpu"
    if requested not in ("cpu", "mps"):
        _note(hw, "Unknown device '%s' (use auto, cpu, cuda, cuda:N or mps), using CPU." % requested)
        return "cpu"
    return requested


def check_model(card, device, hw):
    """Return (device_to_use, warnings) for running `card` on `device`."""
    warnings = []
    if device.startswith("cuda"):
        gpu = next(g for g in hw.gpus if g.index == int(device.split(":")[1]))
        if gpu.free_gb < card.min_vram_gb:
            warnings.append("%s needs about %.1f GB free GPU memory, %s has %.1f GB free -> using CPU instead."
                            % (card.name, card.min_vram_gb, gpu.name, gpu.free_gb))
            device = "cpu"
    if device in ("cpu", "mps") and hw.ram_free_gb < card.min_ram_gb:      # Apple GPUs share the normal RAM
        warnings.append("%s needs about %.1f GB free RAM, only %.1f GB is free. It may be very slow or fail; close "
                        "other programs or pick a smaller model." % (card.name, card.min_ram_gb, hw.ram_free_gb))
    return device, warnings


def describe_device(device, hw):
    if device.startswith("cuda"):
        g = next(g for g in hw.gpus if g.index == int(device.split(":")[1]))
        return "%s (%s, %.1f GB free of %.1f GB)" % (device, g.name, g.free_gb, g.total_gb)
    if device == "mps":
        return "mps (Apple GPU)"
    return "cpu (%s, %d cores)" % (hw.cpu, hw.cpu_cores)


def report(hw, cards=()):
    lines = ["OS:       %s" % hw.os,
             "CPU:      %s, %d cores" % (hw.cpu, hw.cpu_cores),
             "RAM:      %.1f GB free of %.1f GB" % (hw.ram_free_gb, hw.ram_total_gb),
             "PyTorch:  %s (%s)" % (hw.torch_version, "CUDA %s build" % hw.torch_cuda if hw.torch_cuda else "CPU build")]
    for g in hw.gpus:
        lines.append("GPU %d:    %s, %.1f GB free of %.1f GB (%s)" % (g.index, g.name, g.free_gb, g.total_gb,
                                                                   g.capability))
    if hw.mps:
        lines.append("GPU:      Apple Silicon (mps)")
    if not hw.gpus and not hw.mps:
        lines.append("GPU:      none usable -> models run on CPU (slower, but they work)")
    device = pick_device("auto", hw)
    lines.append("Default device: %s" % describe_device(device, hw))
    if cards:
        lines.append("")
        lines.append("Models on %s:" % device)
        for c in cards:
            dev, warns = check_model(c, device, hw)
            status = "OK" if not warns else "! " + " ".join(warns)
            if c.gated:
                from .weights import hf_token, is_downloaded
                status += " | GATED: " + ("downloaded" if is_downloaded(c) else "token found (access must be "
                                          "approved)" if hf_token() else "no Hugging Face token, see "
                                          "docs/GATED_MODELS.md")
            lines.append("  %-16s %s" % (c.name, status))
    for n in hw.notes:
        lines.append("")
        lines.append("NOTE: " + n)
    return "\n".join(lines)
