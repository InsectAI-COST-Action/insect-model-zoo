<div align="center">

# 🐝 InsectAI Model Zoo

**Ready-to-run AI models that find insects in images and say what they are: one install, one command, or a small UI.**

[![Model database](https://img.shields.io/badge/InsectAI-Model%20database-F59E0B)](https://insectai-cost-action.github.io/model-db/)
[![Benchmark database](https://img.shields.io/badge/InsectAI-Benchmark%20database-0EA5E9)](https://insectai-cost-action.github.io/benchmark-dataset-db/)
[![InsectAI](https://img.shields.io/badge/COST%20Action-CA22129%20InsectAI-2E7D32)](https://insectai.eu/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![tests](https://github.com/InsectAI-COST-Action/insect-model-zoo/actions/workflows/tests.yml/badge.svg)](https://github.com/InsectAI-COST-Action/insect-model-zoo/actions/workflows/tests.yml)

<table>
  <tr>
    <td align="center" width="260">
      <a href="https://insectai-cost-action.github.io/model-db/">
        <img src="https://insectai-cost-action.github.io/model-db/icon.svg" width="56" alt="Model database"><br>
        <b>Model database</b>
      </a>
    </td>
    <td align="center" width="260">
      <a href="https://insectai-cost-action.github.io/benchmark-dataset-db/">
        <img src="https://insectai-cost-action.github.io/benchmark-dataset-db/icon.svg" width="56" alt="Benchmark database"><br>
        <b>Benchmark database</b>
      </a>
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
| **Detector** | finds the insects (box, sometimes an outline or a turned box) | insectDCT v8, flat-bug, ArthroNat, Mothbot, Grounding DINO, SAM 3 |
| **Classifier** *(optional)* | says what each insect is (species / family / order + score) | insectDCT classifier V7, BioCLIP 2.5, BioCLIP 2 |

The zoo builds on the InsectAI COST Action's two shared databases: the [**model database**](https://insectai-cost-action.github.io/model-db/) and the
[**benchmark database**](https://insectai-cost-action.github.io/benchmark-dataset-db/); every model in the zoo links to
its page in the model database.

Detectors that come with their own classifier use it by default (insectDCT detector → insectDCT classifier), but any
detector works with any classifier, e.g. flat-bug + BioCLIP 2.5, or SAM 3 + insectDCT classifier.

**Contents:** [Prerequisites](#prerequisites) · [Installation](#installation) · [How to run](#how-to-run) ·
[Models](#models) · [More documentation](#more-documentation) · [Credits and citation](#credits-and-citation)

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
- **Windows**: keep the project in a **short folder that is not synced by OneDrive**, e.g. `C:\code\insect-model-zoo`.
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

Copy the block for your system into a terminal (Windows: *PowerShell*, e.g. the VS Code terminal; macOS: *Terminal*).
It downloads the zoo, creates a virtual environment `.venv` (a private Python folder for this project, so nothing
clashes with other projects), installs everything and checks your hardware.

### Windows (PowerShell)

```powershell
git clone https://github.com/InsectAI-COST-Action/insect-model-zoo.git
cd insect-model-zoo
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # lets this window run the activate script
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --check
```

> [!TIP]
> **"Activate.ps1 cannot be loaded because running scripts is disabled on this system"?**
> Windows PowerShell blocks activation scripts by default. Running `activate.bat` does not help either: in PowerShell
> it runs in a separate cmd process, so nothing gets activated in your window. The quickest fix only affects this one
> terminal window and changes no system settings. Run these two lines:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
>
> Your prompt should then start with `(.venv)`. You need the first line again in every new terminal. To stop the
> error for good for your user account, run this once instead (a work PC may block it):
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Optional: NVIDIA GPU on Windows

The steps above install the CPU version of PyTorch on Windows: everything works, just slower. If your PC has an
**NVIDIA** graphics card, switch to the GPU version (skip this without an NVIDIA card: it only downloads ~2.5 GB for
nothing). **1.** See which card and driver you have:

```powershell
nvidia-smi
```

**2.** Pick the build: **`cu130`** if it shows *CUDA Version 13.0* or higher and the card is an RTX 20-series / GTX 16-series
or newer; otherwise (older driver, or a GTX 10-series or older card) **`cu126`**. If `nvidia-smi` is not found, install
or update the [NVIDIA driver](https://www.nvidia.com/en-us/drivers/) first. **3.** Swap PyTorch (with `.venv` active;
write `cu126` instead of `cu130` if that is your pick) and check that the GPU is found:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python main.py --check
```

### macOS / Linux

```bash
git clone https://github.com/InsectAI-COST-Action/insect-model-zoo.git
cd insect-model-zoo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # Linux WITHOUT an NVIDIA GPU only: saves several GB
python -m pip install -r requirements.txt
python main.py --check
```

GPU: on **Linux** the normal install already includes NVIDIA GPU support (with an older driver, do the same swap as
above with `cu126`); on a **Mac** with Apple Silicon the GPU is used automatically.

### Weights on another drive

The model weights go to `weights/` in the zoo folder (BioCLIP 2.5 alone is 6.8 GB). To keep them on a drive with more
room, e.g. an external SSD, set `WEIGHTS_DIR = "D:/insect-zoo-weights"` at the top of `main.py` (or use
`--weights_dir`, or the environment variable `INSECT_ZOO_WEIGHTS`). `python main.py --check` shows the folder, its free
space, and which models are already downloaded.

### Every time you open a new terminal

Your prompt starts with `(.venv)` when the environment is active. In a new terminal, activate it again before running
the zoo: Windows `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then `.\.venv\Scripts\Activate.ps1`;
macOS / Linux `source .venv/bin/activate`. In VS Code: *Ctrl+Shift+P → "Python: Select Interpreter" → .venv*.

---

## How to run

### The UI

```bash
python main.py
```

A page opens in your browser (http://127.0.0.1:7860) with just an **image**, a **Detector** and a **Classifier**
dropdown, two **confidence** sliders and **Detect**. A sample image is preloaded (four ways insects are
photographed: a light trap, a lab, a citizen-science photo and a camera trap; credits in
[images/CREDITS.md](images/CREDITS.md)), so you can press Detect straight away; click it or drag your own photo onto
it to analyse yours.

- **Detection confidence**: boxes the detector is less sure about are dropped. **Classification confidence**: for
  BioCLIP without names, how sure the answer must be (default 0.5): lower gives more specific names (species), higher
  surer ones (genus, family, order). For the other classifiers, names below it are shown as *Unsure* (default 0).
  Each slider shows its value live.
- **Classifier only**: pick *whole image (classifier only)* as the detector. The whole photo then counts as one box
  and only the classifier runs, e.g. for close-ups where the insect fills the frame (on a wide scene it names the
  whole scene, not one small insect). Results go to `output/whole-image+<classifier>/`.
- Each model family appears once in the lists, with its tags (e.g. *flat-bug (detector + segmentation)*). Families
  with several sizes (insectDCT v8, flat-bug) show size buttons when picked, e.g. N · S · M · L · M v2; the largest
  is selected by default.
- Picking a detector also picks its own classifier (insectDCT → `insectdct-cls-v7`, others → `none`); choose any
  other classifier if you like.
- The text-prompt field unlocks only for detectors that take text (`sam3`: e.g. `bee`), and a names field appears
  for zero-shot classifiers (`bioclip-2.5`, `bioclip-2`: e.g. `Apis mellifera, Bombus terrestris`).
- The first time you use a model, the Detect button shows its download (e.g. "Downloading flatbug-l · 32 MB / 80 MB")
  until it is done.
- The result shows what was found (e.g. "1 found · Apis mellifera ×1 · 1.8 s · cuda:0") and can be downloaded; the
  CSV, COCO and ISIR files are saved to `output/<detector>+<classifier>/` (see [Output](#output)).

Stop the UI with **Ctrl+C** in the terminal.

### The command line

Any argument switches to command-line mode:

```bash
python main.py --list_models                                  # which detectors and classifiers are there
python main.py -i images/test_image.jpg                       # default detector + its classifier, one image
python main.py -m mothbot-mbd-1-1 -c none                     # oriented boxes, on the 4-domain test image
python main.py -m flatbug-m -f path/to/my_images              # every image in a folder
python main.py -m flatbug-m -c insectdct-cls-v7               # flat-bug finds, insectDCT's classifier names
python main.py -m flatbug-m -c bioclip-2.5 --classes "Apis mellifera, Bombus terrestris, Eristalis tenax"
python main.py -m insectdct-v8-s -c none -t 0.25 -d cpu       # detection only, own threshold, force CPU
python main.py -m sam3 -p "bee, butterfly" -c bioclip-2.5     # text prompt (sam3 is gated, see below)
python main.py -m grounding-dino-base -p "bee, moth, fly" -c none   # text prompt, boxes only, not gated
python main.py -f my_camera_1 --camtrapdp --latitude 56.16 --longitude 10.20   # + a Camtrap DP data package
python main.py --download all                                 # fetch all weights now (e.g. before going offline)
python main.py --help                                         # all options + the model list
```

| Argument | Short | What it does | Default |
|---|---|---|---|
| `--model` | `-m` | detector (a wrong name lists the models and suggests the closest one) | `insectdct-v8-m` |
| `--classifier` | `-c` | classifier: `auto` = the detector's own (if any), `none`, or a name | `auto` |
| `--threshold` | `-t` | detection confidence 0–1; lower finds more but also more false positives | the model's own value |
| `--cls_threshold` | | classification confidence 0–1: BioCLIP without names answers at the deepest rank at least this sure (lower = more specific, e.g. `0` = always the species); other classifiers: below it = *Unsure* | BioCLIP: 0.5 |
| `--iou` | | overlap (0–1) above which two boxes are merged as the same insect | the model's own value |
| `--prompt` | `-p` | what to look for, for text-prompt detectors (`sam3`): `"bee"` or `"bee, butterfly"` | `insect` |
| `--classes` | | Latin names for zero-shot classifiers (`bioclip-*`): `"Apis mellifera, Bombus terrestris"` or a `.txt` file with one name per line | every insect species BioCLIP knows |
| `--input_image` | `-i` | one image | `images/test_4_domains.jpg` |
| `--input_folder` | `-f` | all images in a folder (`.jpg .png .tif .bmp .webp .heic`) | – |
| `--output_dir` | `-o` | where results go (a sub-folder per detector + classifier) | `output` |
| `--weights_dir` | | where model weights are kept, e.g. on another drive | `weights/` |
| `--device` | `-d` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps` | `auto` |
| `--camtrapdp` | | also write a [Camtrap DP](docs/CAMTRAP_DP.md) data package (also `--camtrapDP`) | off |
| `--latitude`, `--longitude` | | where the camera was, decimal degrees (WGS84), for ISIR and Camtrap DP | the photos' GPS |
| `--deployment_id` | | Camtrap DP: camera / site name | the images folder's name |
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
CLS_THRESHOLD = None                     # classification confidence 0-1 (None = the classifier's own choice; BioCLIP
                                         # without names: 0.5). BioCLIP answers at the deepest rank at least this sure:
                                         # lower = more specific (species), higher = surer (genus, family, order)
IOU = None                               # overlap above which two boxes count as the same insect; None = model default
PROMPT = None                            # what text-prompt detectors (sam3) look for, e.g. "bee" or "bee, butterfly"
CLASSES = None                           # names for zero-shot classifiers (bioclip): "Apis mellifera, Bombus terrestris"
                                         # or a .txt file with one name per line; None = every species it knows
BIOCLIP_INSECT_TAXA = True               # BioCLIP without your own names: True = picks from every insect species it
                                         # knows (~250,000); False = the whole tree of life (~800,000: plants, birds...)
INPUT_IMAGE = "images/test_4_domains.jpg"  # one image (a light trap, lab, field + camera-trap mosaic) ...
INPUT_FOLDER = None                      # ... or a folder, e.g. "images" (used instead of INPUT_IMAGE when set)
OUTPUT_DIR = "output"                    # results go to OUTPUT_DIR/<detector>[+<classifier>]/
WEIGHTS_DIR = None                       # where model weights are kept; None = weights/ next to this file. Another
                                         # drive with more room: e.g. "D:/insect-zoo-weights" (or --weights_dir)
DEVICE = "auto"                          # auto = NVIDIA GPU (cuda) -> Apple GPU (mps) -> CPU; or "cpu", "cuda:1", ...
CAMTRAPDP = False                        # True = also write a Camtrap DP data package (the camera-trap data standard,
                                         # e.g. for GBIF) to OUTPUT_DIR/.../camtrap-dp/, same as --camtrapdp
CAMTRAPDP_INFO = dict(                   # what Camtrap DP needs to know; check it before you share a package
    project="Insect camera trap",        #   project title
    contributor="",                      #   your name or organisation (contact); the zoo is listed as well
    deployment_id=None,                  #   camera / site name; None = the images folder's name (or --deployment_id)
    latitude=None, longitude=None,       #   camera position in decimal degrees (WGS84); None = the photos' GPS, if any
    capture_method="timeLapse",          #   "timeLapse" (photos at set times) or "activityDetection" (motion trigger)
    sampling_design="targeted",          #   simpleRandom, systematicRandom, clusteredRandom, experimental, targeted
                                         #   or opportunistic
    timezone=None,                       #   the camera clock's zone, used when a photo does not store it: "+02:00" or
                                         #   "Europe/Copenhagen"; None = this computer's zone
    data_license=None,                   #   licence of the data, e.g. "CC-BY-4.0" or "CC0-1.0" (GBIF needs one)
    media_license=None,                  #   licence of the photos, if you share them
)

# Hugging Face token, only needed for GATED models (sam3); how to get one (5 min): docs/GATED_MODELS.md. Best: run
# `hf auth login` once, or put the token in hf_token.txt next to this file (git never uploads that file). Pasting it
# here also works, but git would upload it with main.py, so the zoo warns you.
HF_TOKEN = ""
```

### Output

For each image, in `output/<detector>+<classifier>/` (just `output/<detector>/` without a classifier):

- `<image>_annotated.jpg`: boxes (and outlines for segmentation models) labelled with the taxon and its score, or
  with the detector's label and confidence when there is no classifier
- `<image>_detections.csv`: one row per detection (pixels, top-left origin):
  `image, model, x1, y1, x2, y2, confidence, label, classifier, taxon, taxon_score, taxon_rank, angle`.
  `taxon` is `Unsure` when the insectDCT classifier is not sure at any level.
  `angle` is only filled for oriented boxes (Mothbot): degrees counter-clockwise.
- `<image>_coco.json`: the same in [COCO](https://cocodataset.org/#format-data) format, e.g. to train or fine-tune
  a model on (as pre-labels to check): `bbox` = `[x, y, width, height]` in pixels, the outline of each insect as a
  `segmentation` polygon (segmentation models), and the category = the taxon (or the detector's label when there is
  no classifier or it is unsure). Each annotation also keeps `score`, `label`, `taxon`, `taxon_score`, `taxon_rank`.
- `<image>_isir.json`: the same in **ISIR**, the shared format of the InsectAI model database (boxes with a
  bottom-left origin; the classifier's answers linked by instance id). **[How it is filled, with examples](docs/ISIR.md)**.
- `all_detections.csv`, `coco.json` and `isir.jsonl`: all images of a folder run in one table / one COCO file / one
  ISIR record per line (`file_name` relative to the folder)

### Camtrap DP

With `--camtrapdp`, the zoo also writes a [Camtrap DP](https://camtrap-dp.tdwg.org) data package, the TDWG standard for
camera-trap data (read by GBIF, Agouti, camtraptor): `datapackage.json`, `deployments.csv`, `media.csv` and
`observations.csv`. Give the camera position (`--latitude`, `--longitude`) and check `CAMTRAPDP_INFO` at the top of
`main.py` first. **[How each field is filled, and what GBIF needs](docs/CAMTRAP_DP.md)**.

---

## Models

> ⚠️ <sub>**Licenses are copied from the original authors' work and may not be the full picture.** The insectDCT and
> flat-bug models were trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and the zoo runs
> them (and SAM 3) with Ultralytics code, which is **AGPL-3.0**. Check before commercial use.</sub>

### Detectors (`-m`)

| Model | Tags | Architecture | Weights | Threshold | License | Paper | Code |
|---|---|---|---|---|---|---|---|
| `insectdct-v8-m` | `detector` (+ its classifier by default) | YOLO11m, 1920 px | 39 MB | 0.407 | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `insectdct-v8-s` | `detector` (+ its classifier by default) | YOLO11s, 1920 px | 18 MB | 0.407 | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `flatbug-n` | `detector` `segmentation` | YOLOv8n-seg, 1024 px tiles | 6 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-s` | `detector` `segmentation` | YOLOv8s-seg, 1024 px tiles | 20 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-m` | `detector` `segmentation` | YOLOv8m-seg, 1024 px tiles | 47 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-l` | `detector` `segmentation` | YOLOv8l-seg, 1024 px tiles | 80 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `flatbug-m-v2` | `detector` `segmentation` | YOLO26m-seg, 1024 px tiles | 52 MB | 0.2 | MIT | [MEE](https://doi.org/10.1111/2041-210x.70249) | [flat-bug](https://github.com/darsa-group/flat-bug) |
| `arthronat-l` | `detector` | YOLO11l, 640 px | 49 MB | 0.5 | AGPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.05.06.723207) | [ArthroNat](https://github.com/edgaremy/arthropod-detection-dataset) |
| `arthronat-n` | `detector` | YOLO11n, 640 px | 5 MB | 0.5 | AGPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.05.06.723207) | [ArthroNat](https://github.com/edgaremy/arthropod-detection-dataset) |
| `arthronat-l-mosaic` | `detector` | YOLO11l, 640 px | 49 MB | 0.5 | AGPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.05.06.723207) | [ArthroNat](https://github.com/edgaremy/arthropod-detection-dataset) |
| `arthronat-n-mosaic` | `detector` | YOLO11n, 640 px | 5 MB | 0.5 | AGPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.05.06.723207) | [ArthroNat](https://github.com/edgaremy/arthropod-detection-dataset) |
| `mothbot-mbd-1-1` | `detector` `oriented box` (+ BioCLIP 2 by default) | YOLO26s-OBB, 1600 px | 21 MB | 0.25 | AGPL-3.0 ([see here](docs/MODELS.md#mothbot)) | [bioRxiv](https://doi.org/10.64898/2025.12.03.692171) | [Mothbot](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process) |
| `mothbot-mbd-1-0` | `detector` `oriented box` (+ BioCLIP 2 by default) | YOLO26s-OBB, 1600 px | 21 MB | 0.25 | AGPL-3.0 ([see here](docs/MODELS.md#mothbot)) | [bioRxiv](https://doi.org/10.64898/2025.12.03.692171) | [Mothbot](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process) |
| `grounding-dino-base` | `detector` `text prompt` | Grounding DINO, Swin-B | 890 MB | 0.3 | Apache-2.0 | [ECCV](https://doi.org/10.1007/978-3-031-72970-6_3) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `grounding-dino-tiny` | `detector` `text prompt` | Grounding DINO, Swin-T | 657 MB | 0.3 | Apache-2.0 | [ECCV](https://doi.org/10.1007/978-3-031-72970-6_3) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `sam3` 🔒 **([Gated](docs/GATED_MODELS.md))** | `detector` `segmentation` `text prompt` `gated` | SAM 3, 848M params, 1008 px | 3.2 GB | 0.5 | SAM License | [arXiv](https://doi.org/10.48550/arXiv.2511.16719) | [sam3](https://github.com/facebookresearch/sam3) |

*Threshold* is the default confidence, i.e. the value the authors recommend. Smaller models (s, n) are faster and
need less memory; larger ones are more accurate. 🔒 **Gated** models are free, but you must request access and add a
Hugging Face token once. **[How to (5 minutes)](docs/GATED_MODELS.md)**.

### Classifiers (`-c`)

| Model | Tags | Classes | Architecture | Weights | License | Paper | Code |
|---|---|---|---|---|---|---|---|
| `insectdct-cls-v7` | `classifier` `hierarchical` | 104 insect taxa, as deep as it is sure: order → family → genus/species | ConvNeXt-Base, 224 px crops | 484 MB | GPL-3.0 | [bioRxiv](https://doi.org/10.64898/2026.07.07.736939) | [insectDCT](https://github.com/kimbjerge/insectDCT) |
| `bioclip-2.5` | `classifier` `zero-shot` | **any names you give**; none given: **248,369 insect species** (or the whole tree of life) | ViT-H/14 | 3.7 GB + 3.3 GB species table | MIT | [arXiv](https://doi.org/10.48550/arXiv.2505.23883) | [BioCLIP 2](https://github.com/Imageomics/bioclip-2) |
| `bioclip-2` | `classifier` `zero-shot` | **any names you give**; none given: **264,036 insect species** (or the whole tree of life) | ViT-L/14 | 1.6 GB + 2.7 GB species table | MIT | [arXiv](https://doi.org/10.48550/arXiv.2505.23883) | [BioCLIP 2](https://github.com/Imageomics/bioclip-2) |

Each detected insect is cropped and classified; the result gets a `taxon`, a `taxon_score` (0–1) and, where known,
a `taxon_rank`. Any detector works with any classifier.

**More about each model** (what it is for, authors, paper, weights, licence): [insectDCT detector](docs/MODELS.md#insectdct-detector-v8) ·
[flat-bug](docs/MODELS.md#flat-bug) · [ArthroNat](docs/MODELS.md#arthronat) · [Mothbot](docs/MODELS.md#mothbot) ·
[Grounding DINO](docs/MODELS.md#grounding-dino) · [SAM 3](docs/MODELS.md#sam-3-gated) ·
[insectDCT classifier](docs/MODELS.md#insectdct-hierarchical-classifier-v7) ·
[BioCLIP](docs/MODELS.md#bioclip-25-and-bioclip-2-zero-shot)

---

## More documentation

| Page | What it covers |
|---|---|
| [Models in detail](docs/MODELS.md) | each model: what it is for, authors, paper, code, weights, licence |
| [ISIR output](docs/ISIR.md) | the model database's shared format: how each key is filled, validated examples |
| [Camtrap DP export](docs/CAMTRAP_DP.md) | how the camera-trap data package is filled, time zones, GBIF |
| [How weights are downloaded](docs/WEIGHTS.md) | sources, checksums, offline use, where the weights are kept |
| [GPU / CPU and hardware check](docs/HARDWARE.md) | device choice, CPU fallback, `--check`, getting the GPU to work |
| [Gated models (SAM 3)](docs/GATED_MODELS.md) | access and your Hugging Face token, step by step |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | common errors and how to fix them |
| [Adding a model](docs/ADDING_A_MODEL.md) | registry, model code, tests, file layout |

---

## Credits and citation

The models in this zoo are the work of their authors: **if you use a model, cite its paper.**

**Models**

- **insectDCT** (detector v8, classifier V7): Bjerge, K., Wogram, S. F. A., Serra-Marin, P. E., Sakhiashvili, O., &
  Høye, T. T. (2026). InsectDCT: A generalized pipeline for detection, taxonomic classification, and tracking of insects
  in camera-trap recordings. *bioRxiv*. https://doi.org/10.64898/2026.07.07.736939
- **flat-bug:** Svenning, A., Mougeot, G., Alison, J., Chevalier, D., Chavez Molina, N., Ong, S.-Q., Bjerge, K.,
  Carrillo, J., Høye, T. T., & Geissmann, Q. (2026). A general method for detection and segmentation of terrestrial
  arthropods in images. *Methods in Ecology and Evolution*, 17(3), 727–739. https://doi.org/10.1111/2041-210x.70249
- **ArthroNat:** Remy, E., Carlier, A., Massol, E., Kacimi, R., Chaine, A. S., & Cauchoix, M. (2026). Towards a
  general Detector of terrestrial Arthropods in Natural backgrounds. *bioRxiv*. https://doi.org/10.64898/2026.05.06.723207
- **Mothbot:** Szczygieł, H. A., Johns, B., Fortet, B., Dent, D. H., Quitmeyer, K., & Quitmeyer, A. (2025). Mothbox
  and Mothbot: automated light trap and data processing system for scalable insect monitoring. *bioRxiv*.
  https://doi.org/10.64898/2025.12.03.692171
- **Grounding DINO:** Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H.,
  Zhu, J., & Zhang, L. (2024). Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection.
  In *Computer Vision – ECCV 2024* (pp. 38–55). Springer. https://doi.org/10.1007/978-3-031-72970-6_3
- **SAM 3:** Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). SAM 3: Segment Anything with Concepts.
  *arXiv:2511.16719*. https://doi.org/10.48550/arXiv.2511.16719
- **BioCLIP 2 and 2.5:** Gu, J., Stevens, S., Campolongo, E. G., et al. (2025). BioCLIP 2: Emergent Properties from
  Scaling Hierarchical Contrastive Learning. *arXiv:2505.23883*. https://doi.org/10.48550/arXiv.2505.23883 (BioCLIP
  2.5: see its [model card](https://huggingface.co/imageomics/bioclip-2.5-vith14)). The species tables BioCLIP picks
  from come from the same authors' TreeOfLife-200M dataset.

**Data and standards**

- **Camtrap DP:** Bubnicki, J. W., Norton, B., Baskauf, S. J., et al. (2023). Camtrap DP: an open standard for the FAIR
  exchange and archiving of camera trap data. *Remote Sensing in Ecology and Conservation*.
  https://doi.org/10.1002/rse2.374
- **Test images:** the four photos in `test_4_domains.jpg` come from the AMI trial on Barro Colorado Island (August et
  al., 2023), BIOSCAN-5M (Gharaee et al., 2024), iNaturalist (Andreas Stiller) and the insect camera-trap benchmark of
  Bjerge et al. (2022), all CC BY: full credits in [images/CREDITS.md](images/CREDITS.md).
- The InsectAI [model database](https://insectai-cost-action.github.io/model-db/) and
  [benchmark database](https://insectai-cost-action.github.io/benchmark-dataset-db/).

**Software:** the zoo runs the models with [Ultralytics](https://github.com/ultralytics/ultralytics),
[PyTorch](https://pytorch.org), Hugging Face [transformers](https://github.com/huggingface/transformers),
[OpenCLIP](https://github.com/mlfoundations/open_clip), the [flat-bug](https://github.com/darsa-group/flat-bug) package
and [Gradio](https://www.gradio.app).

**This zoo:** if it helped your work, please cite it too (GitHub's *Cite this repository* button uses
[CITATION.cff](CITATION.cff)):

> Markoff, H., & InsectAI COST Action CA22129 contributors (2026). *InsectAI Model Zoo* [Computer software].
> https://github.com/InsectAI-COST-Action/insect-model-zoo

This repository is based upon work from COST Action CA22129 InsectAI, supported by COST (European Cooperation in
Science and Technology), [www.cost.eu](https://www.cost.eu/).
