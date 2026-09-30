# Models in detail

[← back to the README](../README.md)

What each model in the zoo is for, who made it, where the paper, code and weights are, and its licence. The
overview tables are in the [README](../README.md#models).

> ⚠️ <sub>**Licenses are copied from the original authors' work and may not be the full picture.** Several models were
> trained with and run on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is **AGPL-3.0**. Check
> before commercial use.</sub>

## insectDCT detector (v8)

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

## flat-bug

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

## ArthroNat

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

## Mothbot

Mothbot Detect (MBD), by Digital Naturalism Laboratories: the detection step of Mothbot, the software of the
open-hardware Mothbox light trap. It finds every insect (and spider) on the lit sheet of a light trap and gives each an
**oriented box**, turned to fit its body. The zoo keeps its 4 corners (the outline in the COCO output), its upright
hull (`x1, y1, x2, y2` in the CSV and in Camtrap DP) and its angle (`angle` column: the long side against the image's
x-axis, in degrees counter-clockwise). One class (labelled *insect* here; upstream says *creature*). It runs at 1600 px, handles
thousands of insects per photo, and by default hands them to BioCLIP 2, as Mothbot itself does. MBD-1-1 is the newest
model, MBD-1-0 the previous one; both are YOLO26s-OBB (21 MB).

- **Model database:** [mothbot](https://insectai-cost-action.github.io/model-db/models/mothbot/)
- **Paper:** Szczygieł, H. A., Johns, B., Fortet, B., Dent, D. H., Quitmeyer, K., & Quitmeyer, A. (2025). *Mothbox
  and Mothbot: automated light trap and data processing system for scalable insect monitoring*. bioRxiv. DOI
  [10.64898/2025.12.03.692171](https://doi.org/10.64898/2025.12.03.692171)
- **Code:** [github.com/Digital-Naturalism-Laboratories/Mothbot_Process](https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process)
  · **Weights:** `trained_models/MBD-1-{1,0}.pt` from that repository, pinned to a commit
- **License:** the repository has **no licence file**. The model database lists AGPL-3.0, the licence of Ultralytics,
  which trained it. The zoo only downloads the weights from the authors; ask them before redistributing them.

## Grounding DINO

Open-vocabulary detector from IDEA Research: **type what to look for** (e.g. `insect`, or `bee, moth, fly`) and it
boxes every match, each box labelled with the word it matched best. Like SAM 3 it takes a text prompt, but it gives
boxes only (no outlines), is much smaller (Tiny 657 MB, Base 890 MB) and is **not gated**. It is not trained on
insects specifically: a general word such as `insect` usually finds more than specific names, and it can mix up
look-alikes (e.g. call a fly a moth). The zoo scores each word on its own (the mean of its tokens' scores) and merges
overlapping boxes, since the model does no NMS itself. Default confidence 0.3.

- **Model database:** [grounding-dino](https://insectai-cost-action.github.io/model-db/models/grounding-dino/)
- **Paper:** Liu, S., Zeng, Z., Ren, T., et al. (2024). *Grounding DINO: Marrying DINO with Grounded Pre-training for
  Open-Set Object Detection*. In Computer Vision – ECCV 2024 (pp. 38–55). Springer. DOI
  [10.1007/978-3-031-72970-6_3](https://doi.org/10.1007/978-3-031-72970-6_3) (preprint: [arXiv:2303.05499](https://doi.org/10.48550/arXiv.2303.05499))
- **Code:** [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO); run here with
  Hugging Face `transformers` · **Weights:** [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)
  and [-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) on Hugging Face (open, pinned to a commit)
  · **License:** Apache-2.0

## SAM 3 (gated)

Meta's *Segment Anything Model 3* finds and outlines **whatever you describe in words**: `bee`, `butterfly`,
`ladybird`, or several at once (`bee, butterfly, beetle`), each result labelled with its word. It is a general
foundation model, not trained on insects specifically, so it is great for exploring and for classes no other model
covers. Large (3.2 GB): a GPU with 6 GB+ gives a few seconds per image; on CPU it works, at 40-80 s per image.

- **Model database:** [sam3](https://insectai-cost-action.github.io/model-db/models/sam3/)
- **Access:** 🔒 gated on Hugging Face. Request access and add your token once: **[step-by-step guide](GATED_MODELS.md)**
- **Paper:** Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). *SAM 3: Segment Anything with Concepts*.
  arXiv:2511.16719. DOI [10.48550/arXiv.2511.16719](https://doi.org/10.48550/arXiv.2511.16719)
- **Code:** [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3) · **Weights:**
  [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) (`sam3.pt`, pinned to commit `3c879f3`) ·
  **License:** [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- Runs through the SAM 3 implementation in [Ultralytics](https://docs.ultralytics.com/models/sam-3).

## insectDCT hierarchical classifier (V7)

The classification stage of InsectDCT, used by default with the insectDCT detectors. It classifies each insect to
**order → family → genus/species** and goes only as deep as it is sure (per-class thresholds from the authors);
otherwise the result is `Unsure`. 104 taxa at the deepest level
([list](https://github.com/kimbjerge/insectDCT/blob/main/hierarchicalB3L/datasetV7.txt)), trained on camera-trap crops
of flower visitors. `taxon_score` is the model's probability for that taxon at that level.

- **Model database:** [insectdct](https://insectai-cost-action.github.io/model-db/models/insectdct/)
- **Paper, authors, license:** as the insectDCT detector above (GPL-3.0)
- **Weights:** `HierarchicalClassifierV7` (ConvNeXt-Base) from the Google Drive link in the upstream README, plus the
  upstream classifier code (`common/*.py`) pinned to commit `e459ae8`, both downloaded on first use

## BioCLIP 2.5 and BioCLIP 2 (zero-shot)

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
