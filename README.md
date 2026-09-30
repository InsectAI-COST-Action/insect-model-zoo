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
      </a><br>
      <sub>every insect model, with its paper, licence and weights</sub>
    </td>
    <td align="center" width="260">
      <a href="https://insectai-cost-action.github.io/benchmark-dataset-db/">
        <img src="https://insectai-cost-action.github.io/benchmark-dataset-db/icon.svg" width="56" alt="Benchmark database"><br>
        <b>Benchmark database</b>
      </a><br>
      <sub>datasets to test and compare models on</sub>
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

The zoo builds on the InsectAI COST Action's two shared databases: the [**model database**](https://insectai-cost-action.github.io/model-db/) (the models for
insect detection, classification and traits, each with its paper, licence and weights; every model in the zoo links to
its page there) and the [**benchmark database**](https://insectai-cost-action.github.io/benchmark-dataset-db/) (datasets to test and compare models on).

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
  CSV and COCO JSON files are saved to `output/<detector>+<classifier>/` (see [Output](#output)).

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
| `--camtrapdp` | | also write a [Camtrap DP](#camtrap-dp) data package (also `--camtrapDP`) | off |
| `--latitude`, `--longitude` | | Camtrap DP: where the camera was, decimal degrees (WGS84) | the photos' GPS |
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
- `all_detections.csv` and `coco.json`: all images of a folder run in one table / one COCO file (`file_name` relative
  to the folder)

### Camtrap DP

With `--camtrapdp` (or `CAMTRAPDP = True`), the zoo also writes a
[Camtrap DP 1.0.2](https://camtrap-dp.tdwg.org) data package to `output/<detector>+<classifier>/camtrap-dp/`: the
TDWG standard for camera-trap data, read by GBIF, Agouti, camtraptor (R) and others. It holds `datapackage.json`
(metadata) and `deployments.csv`, `media.csv`, `observations.csv`.

- **One deployment** (camera / site) per run, named after the images folder (or `--deployment_id`). Its position comes
  from `--latitude` / `--longitude` (or `CAMTRAPDP_INFO`), else from the photos' GPS. Without a position the package is
  written but is **not valid yet**: the terminal says so, and you can fill in `latitude` / `longitude` in
  `deployments.csv`.
- **Time** of each photo: its EXIF date, with the time zone the camera stored. Most cameras store none: then
  `timezone` in `CAMTRAPDP_INFO` is used (e.g. `"+02:00"` or `"Europe/Copenhagen"`), else this computer's zone, and
  `mediaComments` says it was assumed. Without an EXIF date: the file's modification time, and the deployment is
  marked `timestampIssues`. The deployment's start / end are the first / last photo.
- **One observation per detection** (`observationLevel` = media), with the box as fractions of the image
  (`bboxX, bboxY, bboxWidth, bboxHeight`), `classificationMethod` = machine, `classifiedBy` = the models used and
  `classificationProbability` = the classifier's score (or the detector's when there is no classifier). A photo
  without detections gets one `blank` observation.
- **Scientific names** only: BioCLIP's are already scientific; insectDCT's own class names are converted (e.g.
  *Aranaea* → Araneae, *Birds* → Aves, *Hymenoptera_bees* → Hymenoptera with the comment "bee"), and boxes it calls
  *Vegetation* are left out. Without a classifier: Insecta (insectDCT) or Arthropoda (flat-bug).
- `filePath` links to each photo where it is (`file:///...`): to publish the package, replace them with the photos'
  web addresses (or upload the photos with it, e.g. to Agouti).
- **GBIF:** the package is ready for GBIF's converter (`gbifIngestion` = media-level observations). GBIF also needs a
  contact and a data licence: set `contributor` and `data_license` in `CAMTRAPDP_INFO` (the terminal reminds you).
- **Check `CAMTRAPDP_INFO`** at the top of `main.py` first: project title, your name, and how the camera took photos
  (`timeLapse` or `activityDetection`) and was placed (`targeted`, `opportunistic`, ...). The package is tested
  against the official Camtrap DP schemas.

A draft of a shared *golden* JSON format (from the team's whiteboard; not produced yet) is in
[docs/GOLDEN_FORMAT.md](docs/GOLDEN_FORMAT.md).

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
| `mothbot-mbd-1-1` | `detector` `oriented box` (+ BioCLIP 2 by default) | YOLO26s-OBB, 1600 px | 21 MB | 0.25 | AGPL-3.0 ([see below](#mothbot)) | [Mothbox, bioRxiv](https://doi.org/10.64898/2025.12.03.692171) | [Mothbot](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process) |
| `mothbot-mbd-1-0` | `detector` `oriented box` (+ BioCLIP 2 by default) | YOLO26s-OBB, 1600 px | 21 MB | 0.25 | AGPL-3.0 ([see below](#mothbot)) | [Mothbox, bioRxiv](https://doi.org/10.64898/2025.12.03.692171) | [Mothbot](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process) |
| `grounding-dino-base` | `detector` `text prompt` | Grounding DINO, Swin-B | 890 MB | 0.3 | Apache-2.0 | [ECCV / arXiv](https://doi.org/10.48550/arXiv.2303.05499) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `grounding-dino-tiny` | `detector` `text prompt` | Grounding DINO, Swin-T | 657 MB | 0.3 | Apache-2.0 | [ECCV / arXiv](https://doi.org/10.48550/arXiv.2303.05499) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
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

### insectDCT detector (v8)

Detects insects in **camera-trap images of flowers and vegetation** (colour time-lapse images, resized to 1920 px).
It is the detection stage of the InsectDCT pipeline for detection, hierarchical taxonomic classification and tracking.
"v8" is the newest detector, trained on detection dataset version 8.
`-m` is the most accurate (upstream best-F1 confidence 0.407); `-s` is made for edge devices such as a Raspberry Pi.

- **Model database:** [insectdct](https://insectai-cost-action.github.io/model-db/models/insectdct/)
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

- **Model database:** [flatbug](https://insectai-cost-action.github.io/model-db/models/flatbug/) · training data in the [benchmark database](https://insectai-cost-action.github.io/benchmark-dataset-db/datasets/flatbug-dataset/)
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

### ArthroNat

YOLO11 detector for arthropods in close-up field photos on **natural backgrounds** (vegetation, soil, flowers), by
Remy et al. (IRIT / SETE-CNRS, Toulouse). One class (labelled *insect* here). It is trained on ArthroNat: about 13,600
research-grade iNaturalist photos of French terrestrial arthropods (979 species in 67 orders), in two mixes:
ArthroNat plus the flat-bug dataset (`arthronat-l`, `arthronat-n`, the paper's recommended setup), and ArthroNat alone
with a 3 x 3 mosaic augmentation (`-mosaic`). L is the more accurate one, N is ten times smaller and faster. Default
confidence 0.5, the value the paper evaluates at.

- **Model database:** [arthronat](https://insectai-cost-action.github.io/model-db/models/arthronat/)
- **Paper:** Remy, E., Carlier, A., Massol, E., Kacimi, R., Chaine, A.S. & Cauchoix, M. (2026). *Towards a general
  Detector of terrestrial Arthropods in Natural backgrounds*. bioRxiv. DOI
  [10.64898/2026.05.06.723207](https://doi.org/10.64898/2026.05.06.723207)
- **Code and dataset:** [github.com/edgaremy/arthropod-detection-dataset](https://github.com/edgaremy/arthropod-detection-dataset)
  (MIT) · **Weights:** [huggingface.co/edgaremy/arthropod-detector](https://huggingface.co/edgaremy/arthropod-detector),
  pinned to a commit · **License:** AGPL-3.0 (the current model card; earlier versions said MIT)

### Mothbot

Mothbot Detect (MBD), by Digital Naturalism Laboratories: the detection step of Mothbot, the software of the
open-hardware Mothbox light trap. It finds every insect (and spider) on the lit sheet of a light trap and gives each an
**oriented box**, turned to fit its body. The zoo keeps its 4 corners (the outline in the COCO output), its upright
hull (`x1, y1, x2, y2` in the CSV and in Camtrap DP) and its angle (`angle` column: the long side against the image's
x-axis, in degrees counter-clockwise). One class (labelled *insect* here; upstream says *creature*). It runs at 1600 px, handles
thousands of insects per photo, and by default hands them to BioCLIP 2, as Mothbot itself does. MBD-1-1 is the newest
model, MBD-1-0 the previous one; both are YOLO26s-OBB (21 MB).

- **Model database:** [mothbot](https://insectai-cost-action.github.io/model-db/models/mothbot/)
- **Paper:** none for Mothbot Detect itself. The Mothbox: Szczygieł, Dent & Quitmeyer (2025). *Mothbox: inexpensive,
  lightweight, automated light trap for scalable insect biodiversity monitoring*. bioRxiv. DOI
  [10.64898/2025.12.03.692171](https://doi.org/10.64898/2025.12.03.692171)
- **Code:** [github.com/Digital-Naturalism-Laboratories/Mothbot_Process](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process)
  · **Weights:** `trained_models/MBD-1-{1,0}.pt` from that repository, pinned to a commit
- **License:** the repository has **no licence file**. The model database lists AGPL-3.0, the licence of Ultralytics,
  which trained it. The zoo only downloads the weights from the authors; ask them before redistributing them.

### Grounding DINO

Open-vocabulary detector from IDEA Research: **type what to look for** (e.g. `insect`, or `bee, moth, fly`) and it
boxes every match, each box labelled with the word it matched best. Like SAM 3 it takes a text prompt, but it gives
boxes only (no outlines), is much smaller (Tiny 657 MB, Base 890 MB) and is **not gated**. It is not trained on
insects specifically: a general word such as `insect` usually finds more than specific names, and it can mix up
look-alikes (e.g. call a fly a moth). The zoo scores each word on its own (the mean of its tokens' scores) and merges
overlapping boxes, since the model does no NMS itself. Default confidence 0.3.

- **Model database:** [grounding-dino](https://insectai-cost-action.github.io/model-db/models/grounding-dino/)
- **Paper:** Liu, S., Zeng, Z., Ren, T., et al. (2024). *Grounding DINO: Marrying DINO with Grounded Pre-Training for
  Open-Set Object Detection*. ECCV 2024. DOI [10.48550/arXiv.2303.05499](https://doi.org/10.48550/arXiv.2303.05499)
- **Code:** [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO); run here with
  Hugging Face `transformers` · **Weights:** [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)
  and [-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) on Hugging Face (open, pinned to a commit)
  · **License:** Apache-2.0

### SAM 3 (gated)

Meta's *Segment Anything Model 3* finds and outlines **whatever you describe in words**: `bee`, `butterfly`,
`ladybird`, or several at once (`bee, butterfly, beetle`), each result labelled with its word. It is a general
foundation model, not trained on insects specifically, so it is great for exploring and for classes no other model
covers. Large (3.2 GB): a GPU with 6 GB+ gives a few seconds per image; on CPU it works, at 40-80 s per image.

- **Model database:** [sam3](https://insectai-cost-action.github.io/model-db/models/sam3/)
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

- **Model database:** [insectdct](https://insectai-cost-action.github.io/model-db/models/insectdct/)
- **Paper, authors, license:** as the insectDCT detector above (GPL-3.0)
- **Weights:** `HierarchicalClassifierV7` (ConvNeXt-Base) from the Google Drive link in the upstream README, plus the
  upstream classifier code (`common/*.py`) pinned to commit `e459ae8`, both downloaded on first use

### BioCLIP 2.5 and BioCLIP 2 (zero-shot)

Biology foundation models from Imageomics, trained on the TreeOfLife-200M images. They need no training for your
data: BioCLIP compares each insect with a text for every name and picks the best match. BioCLIP 2.5 Huge is the
strongest; BioCLIP 2 is half the size.

**Without your own names** (the default), BioCLIP picks from **every insect species it knows** (BioCLIP 2 is less
sure than 2.5 on small crops, so it answers at a broader rank more often: e.g. *Hymenoptera* for the test bee, where
2.5 says *Apis mellifera*; a lower classification confidence, e.g. `--cls_threshold 0`, shows its species guess):

| | BioCLIP 2.5 | BioCLIP 2 |
|---|---|---|
| Insect species (class Insecta, 41 orders) | 248,369 | 264,036 |
| Whole tree of life (`BIOCLIP_INSECT_TAXA = False`) | 794,878 | 867,455 |

It answers at the **deepest rank it is sure of**: the species when that is clear (e.g. *Apis mellifera*), otherwise
the genus, family or order (e.g. *Syrphidae*), always with at least 50 % probability. `taxon_rank` in the CSV says
which rank it is. Raise the *Classification confidence* slider (UI) to get broader but surer names.

The species list and its text features come from the BioCLIP authors
([TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M), CC0, pinned to a commit). They are
downloaded with the model (2.7 GB for BioCLIP 2, 3.3 GB for BioCLIP 2.5) and turned once into a smaller cache
(insects only: about 0.4-0.5 GB).

> [!NOTE]
> **Insects only by default.** BioCLIP is not trained or tuned for insects, and with insects only it names every
> detection as some insect, even something that is not one (e.g. a flower the detector picked up), usually with a low
> score. **To pick from the whole tree of life** (plants, fungi, birds, spiders, ...), open `main.py` and change the
> line near the top to
> ```python
> BIOCLIP_INSECT_TAXA = False
> ```
>
> **Your own names** (`--classes` or the names field in the UI) limit the choice to those names, e.g. the pollinators
> of your site. They must be **Latin names**: *Genus species* (`Apis mellifera`, `Bombus terrestris`) or one genus /
> family / order (`Bombus`, `Syrphidae`); upper or lower case does not matter. Anything else (`honey bee`) is refused
> with a message showing the right format. They are compared *together with* the 16 arthropod orders, so one name
> alone (e.g. `Apis`) still gets a real score instead of always 1.00, and a detection that fits none of your names
> gets the closest order instead of being forced into one of your names.

- **Model database:** [bioclip-2](https://insectai-cost-action.github.io/model-db/models/bioclip-2/) (BioCLIP 2.5 has no page yet)
- **Paper:** Gu, Stevens, Campolongo et al. (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical
  Contrastive Learning*. DOI [10.48550/arXiv.2505.23883](https://doi.org/10.48550/arXiv.2505.23883)
  (BioCLIP 2.5 Huge: see its [model card](https://huggingface.co/imageomics/bioclip-2.5-vith14))
- **Code:** [github.com/Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2) · **License:** MIT
- **Weights:** [imageomics/bioclip-2.5-vith14](https://huggingface.co/imageomics/bioclip-2.5-vith14) and
  [imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2) on Hugging Face (open, pinned to a commit)
- **Species table:** [imageomics/TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
  `embeddings/` (CC0, pinned to a commit)

---

## How weights are downloaded

Weights are **not stored in this repository**. The files stay with the original authors, and the zoo stays small.

1. The first time a model is used (UI, command line or `--download`), `main.py` downloads its weight file from the
   original source listed in [`zoo/registry.py`](zoo/registry.py). You see a progress bar in the terminal or the UI.
2. The file is saved to `weights/<model>/`. An interrupted download resumes where it stopped.
3. The **SHA-256 checksum** is checked, so you get exactly the file this zoo was tested with. If the upstream file
   ever changes, you get a clear error instead of silently different results.
4. Every later run uses the local copy, so it **works offline**. To get everything in advance (fieldwork, cluster
   login node, slow connection), run `python main.py --download all`: detectors ≈ 270 MB, classifiers ≈ 11.5 GB
   (insectDCT 484 MB, BioCLIP 2.5 6.8 GB and BioCLIP 2 4.2 GB, each with its species table), plus 3.2 GB for `sam3`
   once you have access (skipped without). Or download just one: `python main.py --download bioclip-2`.
5. If two windows (e.g. the UI and a command line) need the same model at once, the second one waits for the first
   download to finish instead of downloading it twice.
6. **Gated models** (`sam3`) download from Hugging Face with your token: see [docs/GATED_MODELS.md](docs/GATED_MODELS.md).

Delete `weights/<model>/` to download a model again. To keep weights somewhere else (e.g. a shared drive), set the
environment variable `INSECT_ZOO_WEIGHTS=/path/to/folder`. Each model can have several URLs that are tried in order,
so a mirror can be added later without changing anything else.

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
| **Windows + NVIDIA** | the normal install gives the CPU version of PyTorch: see [Optional: NVIDIA GPU on Windows](#optional-nvidia-gpu-on-windows) |
| **Linux + NVIDIA** | nothing: the normal install already includes GPU support (older driver: the same swap with `cu126`) |
| **Mac (Apple Silicon)** | nothing: used automatically (`mps`) |
| **No GPU** | nothing: runs on CPU |

`python main.py --check` shows whether the GPU is used, and tells you when an NVIDIA card is there but PyTorch cannot
use it. Other versions: [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized / opens the Microsoft Store | Python is not on the PATH; see [Prerequisites](#prerequisites) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | see the tip under [Installation](#installation) |
| `error: externally-managed-environment` | the virtual environment is not active: activate `.venv` first |
| `DLL load failed ... The filename or extension is too long` (Windows) | the folder path is too long: move the project to e.g. `C:\code\insect-model-zoo`, delete `.venv` and create it again |
| `ImportError: libGL.so.1` (Linux server / WSL) | `sudo apt install libgl1 libglib2.0-0` |
| Runs on CPU although you have an NVIDIA GPU | `python main.py --check` tells you why; usually the CPU-only PyTorch, see [GPU](#getting-the-gpu-to-work) |
| Download fails | check the internet connection or proxy; the error shows the URL and where to put a manually downloaded file |
| Too slow | use a GPU, or smaller models (`insectdct-v8-s`, `flatbug-n`, `flatbug-s`, `bioclip-2`), or `-c none` |
| Too many / too few detections | raise / lower `--threshold` (or the *Detection confidence* slider in the UI) |
| BioCLIP gives odd names | give it a list that matches what can be in your images (`--classes`, or the names field in the UI) |
| BioCLIP calls a flower or leaf an insect | by default it only picks from insects: set `BIOCLIP_INSECT_TAXA = False` in `main.py` (see [BioCLIP](#bioclip-25-and-bioclip-2-zero-shot)) |
| `sam3`: *No Hugging Face token found* / *error 401* / *error 403* | see [docs/GATED_MODELS.md](docs/GATED_MODELS.md#if-it-does-not-work) |
| UI on a remote Linux server / cluster (no screen) | run `python main.py --ui --port 7860` there and `ssh -L 7860:localhost:7860 you@server` from your laptop, then open http://127.0.0.1:7860; or just use the command line |

---

## Adding a model

1. Add a `ModelCard` to [`zoo/registry.py`](zoo/registry.py): name, download URL(s), file size, SHA-256, default
   threshold, memory needs, paper DOI, license. Detectors can name a `default_classifier`. Use `kind="classifier"`
   for classifiers, `classes=(...)` for zero-shot ones (unlocks the names field and `--classes`),
   `text_prompt=True` for text-prompted detectors (unlocks the prompt field and `--prompt`), and
   `gated="<Hugging Face page>"` for gated models (token handling and help links come free). Give sizes/versions of
   one model the same `group` and their own `variant` (e.g. flat-bug `N`, `S`, `M`): the UI then lists the family once
   and shows the sizes as buttons, with the largest selected by default. Tags are worked out from these fields.
2. If it is a new kind of model, add `zoo/families/<family>.py` (see the existing families):
   - a detector: class `Model(card, weights_path, device)` with `predict(image_rgb, threshold, iou, prompt=None)`
     returning a list of `Detection`;
   - a classifier: class `Classifier(card, weights_path, device)` with `classify(image_rgb, detections, classes=None)`
     that sets `taxon`, `taxon_score` (and `taxon_rank`) on each detection.
3. Add any new packages to `requirements.txt` and a row to the [Models](#models) tables.
   If the model has a page in the [InsectAI model database](https://insectai-cost-action.github.io/model-db/), set `model_db="<page>"` (and its datasets in the
   [benchmark database](https://insectai-cost-action.github.io/benchmark-dataset-db/) as `datasets=(("<page>", "<title>"),)`): the UI, `--list_models` and the exports link
   to them. A model that is not in the model database yet is best added there too.
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
zoo/results.py          drawing and CSV / COCO JSON output per image
zoo/export.py           COCO JSON and Camtrap DP data package
zoo/ui.py               the web UI (Gradio)
images/                test images: test_4_domains.jpg (light trap, lab, citizen science, camera trap; credits
                        in images/CREDITS.md) and test_image.jpg (a honey bee from our own camera)
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
