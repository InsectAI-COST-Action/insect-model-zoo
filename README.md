<div align="center">

# 🐝 InsectAI Model Zoo

**Ready-to-run AI models that find insects in images and say what they are: one install, one command, or a small UI.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-insectai--cost-FFD21E)](https://huggingface.co/insectai-cost)
[![InsectAI](https://img.shields.io/badge/COST%20Action-CA22129%20InsectAI-2E7D32)](https://insectai.eu/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![tests](https://github.com/HugoMarkoff/Insect_model_zoo/actions/workflows/tests.yml/badge.svg)](https://github.com/HugoMarkoff/Insect_model_zoo/actions/workflows/tests.yml)

<table>
  <tr>
    <td align="center" width="260">
      <a href="https://huggingface.co/insectai-cost">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="56" alt="Hugging Face"><br>
        <b>InsectAI on Hugging Face</b>
      </a><br>
      <sub>huggingface.co/insectai-cost</sub>
    </td>
  </tr>
</table>

</div>

---

## About

[**InsectAI**](https://insectai.eu/) is a European COST Action ([CA22129](https://www.cost.eu/actions/CA22129/),
2023–2027). It brings together researchers and stakeholders who use image-based AI for insect monitoring and
conservation, to help understand and counteract the widespread decline of insects.

Many good insect models already exist, but they are scattered across papers, repositories and download links, and each
one has its own setup. **This model zoo collects them in one place**, with one installation, one command line and one
small UI. Ecologists can try the models on their own images, compare them and pick the right one, and every model stays
credited to its original authors.

It works in two steps, and you can combine any two models:

| Step | What it does | Models |
|---|---|---|
| **Detector** | finds the insects (box, sometimes an outline) | insectDCT v8, flat-bug, SAM 3 |
| **Classifier** *(optional)* | says what each insect is (species / family / order + score) | insectDCT classifier V7, BioCLIP 2.5, BioCLIP 2 |

Detectors that come with their own classifier use it by default (insectDCT detector → insectDCT classifier), but any
detector works with any classifier, e.g. flat-bug + BioCLIP 2.5, or SAM 3 + insectDCT classifier.

**Contents:** [Prerequisites](#prerequisites) · [Installation](#installation) · [How to run](#how-to-run) ·
[Models](#models) · [How weights are downloaded](#how-weights-are-downloaded) · [GPU / CPU and hardware check](#gpu--cpu-and-hardware-check) ·
[Troubleshooting](#troubleshooting) · [Adding a model](#adding-a-model) · [Credits](#credits-and-citation)

---

## Prerequisites

Starting from a fresh computer, you need **Git**, **Python 3.11 or newer**, and optionally a code editor. Around
**5 GB of free disk space** (mostly PyTorch) and an internet connection for the first run.

**Works on** Windows 10/11 (64-bit), Linux (x86-64 and ARM64) and Macs with Apple Silicon (M1 or newer, macOS 11+).
It is tested automatically on all three on every change. Intel Macs cannot run it, because current PyTorch no longer
supports them.

| | What it is for | Windows | macOS | Linux |
|---|---|---|---|---|
| **[Git](https://git-scm.com/downloads)** | downloads this repository | [installer](https://git-scm.com/downloads/win) or `winget install Git.Git` | `xcode-select --install` or `brew install git` | usually installed, else `sudo apt install git` |
| **[Python 3.11+](https://www.python.org/downloads/)** | runs the code (3.12 recommended) | [installer](https://www.python.org/downloads/windows/). Tick **"Add python.exe to PATH"** | [installer](https://www.python.org/downloads/macos/) or `brew install python@3.12` | usually installed (check the version!) |
| **[VS Code](https://code.visualstudio.com/)** *(optional)* | editor to change settings in `main.py` | installer + [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | same | same |
| **[NVIDIA driver](https://www.nvidia.com/en-us/drivers/)** *(optional)* | faster with an NVIDIA GPU | only if you have an NVIDIA GPU | not needed: Apple Silicon GPUs are used automatically | only if you have an NVIDIA GPU |

Check that it worked by opening a terminal (Windows: *PowerShell*; macOS: *Terminal*) and typing:

```bash
git --version
python --version      # Windows
python3 --version     # macOS / Linux  -> must say 3.11 or higher
```

<details>
<summary><b>Notes per operating system</b> (things that often go wrong)</summary>

- **Windows**: if `python` opens the Microsoft Store or is "not recognized", Python is not on the PATH. Re-run the
  installer, choose *Modify*, and tick *Add Python to environment variables*.
- **Windows**: keep the project in a **short folder that is not synced by OneDrive**, e.g. `C:\code\Insect_model_zoo`.
  Deep paths (like `OneDrive - ...\Desktop\...`) can break loading some libraries, and OneDrive would try to sync
  thousands of `.venv` files.
- **macOS** does not come with a usable Python 3. Typing `python3` may offer the Xcode Command Line Tools, but their
  Python (3.9) is too old for this project. Install 3.12 from python.org or with Homebrew.
- **Linux**: Python is usually pre-installed, but check the version. Ubuntu 24.04 (3.12) and Debian 12 (3.11) are
  fine; Ubuntu 22.04 has 3.10, which is too old (use e.g. `uv python install 3.12` or the deadsnakes PPA).
  On Ubuntu/Debian also run `sudo apt install python3-venv python3-pip`.

</details>

---

## Installation

**1. Download the repository**

```bash
git clone https://github.com/HugoMarkoff/Insect_model_zoo.git
cd Insect_model_zoo
```

**2. Create and activate a virtual environment** (a private Python folder `.venv` for this project, so nothing clashes
with other projects)

| Windows (PowerShell) | macOS / Linux |
|---|---|
| `python -m venv .venv` | `python3 -m venv .venv` |
| `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

Your prompt now starts with `(.venv)`. **Activate again every time you open a new terminal.** If PowerShell says
*running scripts is disabled*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
In VS Code: *Ctrl+Shift+P → "Python: Select Interpreter" → .venv*.

**3. (NVIDIA GPU on Windows only) install the GPU version of PyTorch first**, see [GPU](#gpu--cpu-and-hardware-check).

**4. Install the requirements**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**5. Check that everything is found**

```bash
python main.py --check
```

---

## How to run

### The UI

```bash
python main.py
```

A page opens in your browser (http://127.0.0.1:7860) with just an **image**, a **Detector** and a **Classifier**
dropdown, a **threshold** slider and **Detect**. The test image is preloaded, so you can press Detect straight away.

- Picking a detector also picks its own classifier (insectDCT → `insectdct-cls-v7`, others → `none`); choose any
  other classifier if you like.
- The text-prompt field unlocks only for detectors that take text (`sam3`: e.g. `bee`), and a names field appears
  for zero-shot classifiers (`bioclip-2.5`, `bioclip-2`: e.g. `Apis mellifera, Bombus terrestris`).
- The first time you use a model, the Detect button shows its download (e.g. "Downloading flatbug-l · 32 MB / 80 MB")
  until it is done.
- The result shows what was found (e.g. "1 found · Apis mellifera ×1 · 1.8 s · cuda:0") and can be downloaded; the
  CSV/JSON files are saved to `output/<detector>+<classifier>/`.

Stop the UI with **Ctrl+C** in the terminal.

### The command line

Any argument switches to command-line mode:

```bash
python main.py --list_models                                  # which detectors and classifiers are there
python main.py -i images/test_image.jpg                       # default detector + its classifier, one image
python main.py -m flatbug-m -f path/to/my_images              # every image in a folder
python main.py -m flatbug-m -c insectdct-cls-v7               # flat-bug finds, insectDCT's classifier names
python main.py -m flatbug-m -c bioclip-2.5 --classes "Apis mellifera, Bombus terrestris, Eristalis tenax"
python main.py -m insectdct-v8-s -c none -t 0.25 -d cpu       # detection only, own threshold, force CPU
python main.py -m sam3 -p "bee, butterfly" -c bioclip-2.5     # text prompt (sam3 is gated, see below)
python main.py --download all                                 # fetch all weights now (e.g. before going offline)
python main.py --help                                         # all options + the model list
```

| Argument | Short | What it does | Default |
|---|---|---|---|
| `--model` | `-m` | detector (a wrong name lists the models and suggests the closest one) | `insectdct-v8-m` |
| `--classifier` | `-c` | classifier: `auto` = the detector's own (if any), `none`, or a name | `auto` |
| `--threshold` | `-t` | detection confidence 0–1; lower finds more but also more false positives | the model's own value |
| `--iou` | | overlap (0–1) above which two boxes are merged as the same insect | the model's own value |
| `--prompt` | `-p` | what to look for, for text-prompt detectors (`sam3`): `"bee"` or `"bee, butterfly"` | `insect` |
| `--classes` | | names for zero-shot classifiers (`bioclip-*`): `"Apis mellifera, Bombus terrestris"` or a `.txt` file with one name per line | arthropod orders |
| `--input_image` | `-i` | one image | `images/test_image.jpg` |
| `--input_folder` | `-f` | all images in a folder (`.jpg .png .tif .bmp .webp .heic`) | – |
| `--output_dir` | `-o` | where results go (a sub-folder per detector + classifier) | `output` |
| `--device` | `-d` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps` | `auto` |
| `--list_models` | `-l` | list the models and exit | |
| `--check` | | hardware report: GPU, RAM, which models fit | |
| `--download MODEL` | | only download weights (`all` = every model) | |
| `--ui` | | open the UI, e.g. `python main.py --ui -m flatbug-s` | |
| `--help` | `-h` | help | |

### Default settings

If you would rather not type arguments, change the defaults at the top of `main.py`:

```python
MODEL = "insectdct-v8-m"                 # detector, see `python main.py --list_models`
CLASSIFIER = "auto"                      # auto = the detector's own classifier (insectDCT -> insectdct-cls-v7, others
                                         # none), "none", or a classifier name, e.g. "bioclip-2.5"
THRESHOLD = None                         # detection confidence 0-1; None = the model's recommended value
IOU = None                               # overlap above which two boxes count as the same insect; None = model default
PROMPT = None                            # what text-prompt detectors (sam3) look for, e.g. "bee" or "bee, butterfly"
CLASSES = None                           # names for zero-shot classifiers (bioclip): "Apis mellifera, Bombus terrestris"
                                         # or a .txt file with one name per line; None = arthropod orders
INPUT_IMAGE = "images/test_image.jpg"    # one image ...
INPUT_FOLDER = None                      # ... or a folder, e.g. "images" (used instead of INPUT_IMAGE when set)
OUTPUT_DIR = "output"                    # results go to OUTPUT_DIR/<detector>[+<classifier>]/
DEVICE = "auto"                          # auto = NVIDIA GPU (cuda) -> Apple GPU (mps) -> CPU; or "cpu", "cuda:1", ...

# Hugging Face token, only needed for GATED models (sam3). Paste it between the quotes: HF_TOKEN = "hf_..."
HF_TOKEN = ""
```

### Output

For each image, in `output/<detector>+<classifier>/` (just `output/<detector>/` without a classifier):

- `<image>_annotated.jpg`: boxes (and outlines for segmentation models) labelled with the taxon and its score, or
  with the detector's label and confidence when there is no classifier
- `<image>_detections.csv`: one row per detection (pixels, top-left origin):
  `image, model, x1, y1, x2, y2, confidence, label, classifier, taxon, taxon_score, taxon_rank`.
  `taxon` is `Unsure` when the insectDCT classifier is not sure at any level.
- `<image>_detections.json`: the same plus the outline polygon of each insect (segmentation models only)
- `all_detections.csv`: all images of a folder run in one table

---

## Models

### Detectors (`-m`)

| Model | Task | Architecture | Weights | Threshold | License | Paper | Code |
|---|---|---|---|---|---|---|---|
| `insectdct-v8-m` | detection (+ its classifier by default) | YOLO11m, 1920 px | 39 MB | 0.407 | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `insectdct-v8-s` | detection (+ its classifier by default) | YOLO11s, 1920 px | 18 MB | 0.407 | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `flatbug-n` | detection + segmentation | YOLOv8n-seg, 1024 px tiles | 6 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-s` | detection + segmentation | YOLOv8s-seg, 1024 px tiles | 20 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-m` | detection + segmentation | YOLOv8m-seg, 1024 px tiles | 47 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-l` | detection + segmentation | YOLOv8l-seg, 1024 px tiles | 80 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-m-v2` | detection + segmentation | YOLO26m-seg, 1024 px tiles | 52 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `sam3` 🔒 **([Gated](docs/GATED_MODELS.md))** | segmentation of what you type | SAM 3, 848M params, 1008 px | 3.2 GB | 0.5 | SAM License | [arXiv](https://doi.org/10.48550/arXiv.2511.16719) | [sam3](https://github.com/facebookresearch/sam3) |

*Threshold* is the default confidence, i.e. the value the authors recommend. Smaller models (s, n) are faster and
need less memory; larger ones are more accurate. 🔒 **Gated** models are free, but you must request access and add a
Hugging Face token once. **[How to (5 minutes)](docs/GATED_MODELS.md)**.

### Classifiers (`-c`)

| Model | Classes | Architecture | Weights | License | Paper | Code |
|---|---|---|---|---|---|---|
| `insectdct-cls-v7` | 104 insect taxa, as deep as it is sure: order → family → genus/species | ConvNeXt-Base, 224 px crops | 484 MB | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `bioclip-2.5` | **any names you give** (zero-shot); default: 16 arthropod orders | ViT-H/14 | 3.7 GB | MIT | [arXiv](https://doi.org/10.48550/arXiv.2505.23883) | [BioCLIP 2](https://github.com/Imageomics/bioclip-2) |
| `bioclip-2` | **any names you give** (zero-shot); default: 16 arthropod orders | ViT-L/14 | 1.6 GB | MIT | [arXiv](https://doi.org/10.48550/arXiv.2505.23883) | [BioCLIP 2](https://github.com/Imageomics/bioclip-2) |

Each detected insect is cropped and classified; the result gets a `taxon`, a `taxon_score` (0–1) and, where known,
a `taxon_rank`. Any detector works with any classifier.

### insectDCT detector (v8)

Detects insects in **camera-trap images of flowers and vegetation** (colour time-lapse images, resized to 1920 px).
It is the detection stage of the InsectDCT pipeline for detection, hierarchical taxonomic classification and tracking.
"v8" is the newest detector, trained on detection dataset version 8.
`-m` is the most accurate (upstream best-F1 confidence 0.407); `-s` is made for edge devices such as a Raspberry Pi.

- **Authors:** Kim Bjerge, Simon F. A. Wogram, Pau Enric Serra-Marin, Otar Sakhiashvili, Toke T. Høye
- **Paper:** *InsectDCT: A generalized pipeline for detection, taxonomic classification, and tracking of insects in
  camera-trap recordings* (2026), bioRxiv. DOI [10.64898/2026.07.07.736939](https://doi.org/10.64898/2026.07.07.736939)
- **Code:** [github.com/kimbjerge/insectDCT](https://github.com/kimbjerge/insectDCT) · **Dataset (V6):**
  [zenodo.org/records/21154490](https://zenodo.org/records/21154490) · **License:** GPL-3.0
- **Weights:** `runs/detect/insects8Color{,11s}/weights/best.pt` from the upstream repository, pinned to commit
  [`e459ae8`](https://github.com/kimbjerge/insectDCT/tree/e459ae87ebe39732963e77e0fcdd0a9a28672a08)
- *Not included yet:* the `Motion` detectors (they need consecutive video frames).

### flat-bug

Detects **and outlines** all terrestrial arthropods on any background, in images of any size: large images are cut
into tiles at several scales (a "pyramid") and the results merged. Tuned especially for top-down images and scans
(hence "flat"). Sizes N/S/M/L are the YOLOv8 models from the paper; `flatbug-m-v2` is the newer YOLO26 model that is
the default since flat-bug 1.2.

- **Authors:** Asger Svenning, Guillaume Mougeot, Jamie Alison, Daphne Chevalier, Nisa Chavez Molina, Song-Quan Ong,
  Kim Bjerge, Juli Carrillo, Toke T. Høye, Quentin Geissmann
- **Paper:** *A general method for detection and segmentation of terrestrial arthropods in images* (2026), Methods in
  Ecology and Evolution 17(3), 727–739. DOI [10.1111/2041-210x.70249](https://doi.org/10.1111/2041-210x.70249)
- **Code:** [github.com/darsa-group/flat-bug](https://github.com/darsa-group/flat-bug) · **Docs:**
  [darsa.info/flat-bug](https://darsa.info/flat-bug/) · **Dataset:**
  [10.5281/zenodo.14761446](https://doi.org/10.5281/zenodo.14761446) · **License:** MIT
- **Weights:** `flat_bug_{N,S,M,L,M_v2}.pt` from the authors' model store at Aarhus University (ERDA), the same source
  the `flat-bug` package uses. The inference code is the official [`flat-bug`](https://pypi.org/project/flat-bug/)
  package from PyPI.

### SAM 3 (gated)

Meta's *Segment Anything Model 3* finds and outlines **whatever you describe in words**: `bee`, `butterfly`,
`ladybird`, or several at once (`bee, butterfly, beetle`), each result labelled with its word. It is a general
foundation model, not trained on insects specifically, so it is great for exploring and for classes no other model
covers. Large (3.2 GB): a GPU with 6 GB+ gives a few seconds per image; on CPU it works, at about 40 s per image.

- **Access:** 🔒 gated on Hugging Face. Request access and add your token once: **[step-by-step guide](docs/GATED_MODELS.md)**
- **Paper:** Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). *SAM 3: Segment Anything with Concepts*.
  arXiv:2511.16719. DOI [10.48550/arXiv.2511.16719](https://doi.org/10.48550/arXiv.2511.16719)
- **Code:** [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3) · **Weights:**
  [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) (`sam3.pt`, pinned to commit `3c879f3`) ·
  **License:** [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- Runs through the SAM 3 implementation in [Ultralytics](https://docs.ultralytics.com/models/sam-3).

### insectDCT hierarchical classifier (V7)

The classification stage of InsectDCT, used by default with the insectDCT detectors. It classifies each insect to
**order → family → genus/species** and goes only as deep as it is sure (per-class thresholds from the authors);
otherwise the result is `Unsure`. 104 taxa at the deepest level
([list](https://github.com/kimbjerge/insectDCT/blob/main/hierarchicalB3L/datasetV7.txt)), trained on camera-trap crops
of flower visitors. `taxon_score` is the model's probability for that taxon at that level.

- **Paper, authors, license:** as the insectDCT detector above (GPL-3.0)
- **Weights:** `HierarchicalClassifierV7` (ConvNeXt-Base) from the Google Drive link in the upstream README, plus the
  upstream classifier code (`common/*.py`) pinned to commit `e459ae8`, both downloaded on first use

### BioCLIP 2.5 and BioCLIP 2 (zero-shot)

Biology foundation models from Imageomics, trained on the TreeOfLife-200M images. **You choose the names** (species,
genera, families, orders, common names) and it picks the best match for each insect, so no training is needed. Without
names it chooses between 16 arthropod orders (Hymenoptera, Diptera, Coleoptera, Lepidoptera, ...). BioCLIP 2.5 Huge is
the strongest; BioCLIP 2 is half the size. Works best with a list that matches what can be in your images, e.g. the
pollinators of your site. `taxon_score` is the probability among the names you gave.

- **Paper:** Gu, Stevens, Campolongo et al. (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical
  Contrastive Learning*. DOI [10.48550/arXiv.2505.23883](https://doi.org/10.48550/arXiv.2505.23883)
  (BioCLIP 2.5 Huge: see its [model card](https://huggingface.co/imageomics/bioclip-2.5-vith14))
- **Code:** [github.com/Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2) · **License:** MIT
- **Weights:** [imageomics/bioclip-2.5-vith14](https://huggingface.co/imageomics/bioclip-2.5-vith14) and
  [imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2) on Hugging Face (open, pinned to a commit)

> **Licensing note:** each model keeps its authors' license (tables above). The insectDCT and flat-bug detectors
> are trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). For commercial use, check
> Ultralytics' licensing too.

---

## How weights are downloaded

Weights are **not stored in this repository**. The files stay with the original authors, and the zoo stays small.

1. The first time a model is used (UI, command line or `--download`), `main.py` downloads its weight file from the
   original source listed in [`zoo/registry.py`](zoo/registry.py). You see a progress bar in the terminal or the UI.
2. The file is saved to `weights/<model>/`. An interrupted download resumes where it stopped.
3. The **SHA-256 checksum** is checked, so you get exactly the file this zoo was tested with. If the upstream file
   ever changes, you get a clear error instead of silently different results.
4. Every later run uses the local copy, so it **works offline**. To get everything in advance (fieldwork, cluster
   login node, slow connection), run `python main.py --download all`: detectors ≈ 270 MB, classifiers ≈ 5.8 GB
   (insectDCT 484 MB, BioCLIP 2.5 3.7 GB, BioCLIP 2 1.6 GB), plus 3.2 GB for `sam3` once you have access (skipped
   without). Or download just one: `python main.py --download bioclip-2`.
6. If two windows (e.g. the UI and a command line) need the same model at once, the second one waits for the first
   download to finish instead of downloading it twice.
5. **Gated models** (`sam3`) download from Hugging Face with your token: see [docs/GATED_MODELS.md](docs/GATED_MODELS.md).

Delete `weights/<model>/` to download a model again. To keep weights somewhere else (e.g. a shared drive), set the
environment variable `INSECT_ZOO_WEIGHTS=/path/to/folder`. Each model can have several URLs that are tried in order,
so a mirror on [Hugging Face](https://huggingface.co/insectai-cost) can be added later without changing anything else.

---

## GPU / CPU and hardware check

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
RAM:      39.8 GB free of 63.4 GB
PyTorch:  2.14.0+cu130 (CUDA 13.0 build)
GPU 0:    NVIDIA GeForce RTX 5090 Laptop GPU, 22.6 GB free of 23.9 GB (sm_120)
Default device: cuda:0

Models on cuda:0:
  insectdct-v8-m   OK
  flatbug-m        OK
  ...
```

### Getting the GPU to work

| System | What to do |
|---|---|
| **Linux + NVIDIA** | nothing: `pip install -r requirements.txt` already installs PyTorch with CUDA |
| **Windows + NVIDIA** | the default PyTorch on Windows is CPU-only. Install the CUDA build **before** the requirements (below) |
| **Mac (Apple Silicon)** | nothing: used automatically (`mps`) |
| **No GPU** | nothing: runs on CPU |

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

`cu130` needs a recent NVIDIA driver (580 or newer) and is required for RTX 50-series cards. With an older driver use `cu126`. Run
`nvidia-smi` to see your driver version, and see [pytorch.org](https://pytorch.org/get-started/locally/) for other
versions. Already installed the CPU version? Run `python -m pip uninstall -y torch torchvision` first, then the two
lines above.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized / opens the Microsoft Store | Python is not on the PATH; see [Prerequisites](#prerequisites) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `error: externally-managed-environment` | the virtual environment is not active: activate `.venv` first |
| `DLL load failed ... The filename or extension is too long` (Windows) | the folder path is too long: move the project to e.g. `C:\code\Insect_model_zoo`, delete `.venv` and create it again |
| `ImportError: libGL.so.1` (Linux server / WSL) | `sudo apt install libgl1 libglib2.0-0` |
| Runs on CPU although you have an NVIDIA GPU | `python main.py --check` tells you why; usually the CPU-only PyTorch, see [GPU](#getting-the-gpu-to-work) |
| Download fails | check the internet connection or proxy; the error shows the URL and where to put a manually downloaded file |
| Too slow | use a GPU, or smaller models (`insectdct-v8-s`, `flatbug-n`, `flatbug-s`, `bioclip-2`), or `-c none` |
| Too many / too few detections | raise / lower `--threshold` (or the slider in the UI) |
| BioCLIP gives odd names | give it a list that matches what can be in your images (`--classes`, or the names field in the UI) |
| `sam3`: *No Hugging Face token found* / *error 401* / *error 403* | see [docs/GATED_MODELS.md](docs/GATED_MODELS.md#if-it-does-not-work) |
| UI on a remote Linux server / cluster (no screen) | run `python main.py --ui --port 7860` there and `ssh -L 7860:localhost:7860 you@server` from your laptop, then open http://127.0.0.1:7860; or just use the command line |

---

## Adding a model

1. Add a `ModelCard` to [`zoo/registry.py`](zoo/registry.py): name, download URL(s), file size, SHA-256, default
   threshold, memory needs, paper DOI, license. Detectors can name a `default_classifier`. Use `kind="classifier"`
   for classifiers, `classes=(...)` for zero-shot ones (unlocks the names field and `--classes`),
   `text_prompt=True` for text-prompted detectors (unlocks the prompt field and `--prompt`), and
   `gated="<Hugging Face page>"` for gated models (token handling and help links come free).
2. If it is a new kind of model, add `zoo/families/<family>.py` (see the existing families):
   - a detector: class `Model(card, weights_path, device)` with `predict(image_rgb, threshold, iou, prompt=None)`
     returning a list of `Detection`;
   - a classifier: class `Classifier(card, weights_path, device)` with `classify(image_rgb, detections, classes=None)`
     that sets `taxon`, `taxon_score` (and `taxon_rank`) on each detection.
3. Add any new packages to `requirements.txt` and a row to the [Models](#models) tables.
4. Test: `python main.py -m <detector> -c <classifier> -d cpu`, and again on GPU without `-d cpu`.

Suggestions and pull requests are welcome, especially from InsectAI members with models to share.

```text
main.py                 settings at the top, command line, starts the UI
zoo/registry.py         the model list (links, DOIs, licenses, checksums)
zoo/families/           the code that runs each kind of model: detectors (insectdct.py, flatbug.py, sam3.py) and
                        classifiers (insectdct_cls.py, bioclip.py)
docs/GATED_MODELS.md    how to get access to gated models (sam3) and where the token goes
zoo/weights.py          download + cache + checksum of weights (one download at a time per file)
zoo/hardware.py         device choice and hardware check
zoo/engine.py           load the detector + classifier and run them (GPU -> CPU fallback)
zoo/results.py          drawing and CSV / JSON output
zoo/ui.py               the web UI (Gradio)
images/test_image.jpg   a test image (a honey bee from our own camera)
assets/                 InsectAI and COST logos for the UI
.github/workflows/      automatic tests on Windows, Linux and macOS
```

---

## Credits and citation

The models in this zoo are the work of their authors, so **if you use a model, cite its paper** (see [Models](#models)):

- Bjerge, K., Wogram, S. F. A., Serra-Marin, P. E., Sakhiashvili, O., & Høye, T. T. (2026). InsectDCT: A generalized
  pipeline for detection, taxonomic classification, and tracking of insects in camera-trap recordings. *bioRxiv*.
  https://doi.org/10.64898/2026.07.07.736939
- Svenning, A., Mougeot, G., Alison, J., Chevalier, D., Chavez Molina, N., Ong, S.-Q., Bjerge, K., Carrillo, J.,
  Høye, T. T., & Geissmann, Q. (2026). A general method for detection and segmentation of terrestrial arthropods in
  images. *Methods in Ecology and Evolution*, 17(3), 727–739. https://doi.org/10.1111/2041-210x.70249
- Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). SAM 3: Segment Anything with Concepts. *arXiv:2511.16719*.
  https://doi.org/10.48550/arXiv.2511.16719
- Gu, J., Stevens, S., Campolongo, E. G., et al. (2025). BioCLIP 2: Emergent Properties from Scaling Hierarchical
  Contrastive Learning. *arXiv:2505.23883*. https://doi.org/10.48550/arXiv.2505.23883

This repository is based upon work from COST Action CA22129 InsectAI, supported by COST (European Cooperation in
Science and Technology), [www.cost.eu](https://www.cost.eu/).
