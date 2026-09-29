"""
The list of models in the zoo. One ModelCard per model; `family` picks the code in zoo/families/ that runs it.

To add a model: add a ModelCard to MODELS (and a family module if it is a new kind of model), then add a row to the
"Models" table in README.md.
"""

import difflib
from dataclasses import dataclass, field

MB = 1024 * 1024
REPO_URL = "https://github.com/HugoMarkoff/Insect_model_zoo"
GATED_GUIDE_URL = REPO_URL + "/blob/main/docs/GATED_MODELS.md"    # how to get access + where to put the token


@dataclass(frozen=True)
class WeightFile:
    filename: str
    urls: tuple            # tried in order, so a mirror (e.g. Hugging Face) can be added in front later
    size: int              # bytes; used for the progress bar and the free-disk check
    sha256: str            # checked after download


@dataclass(frozen=True)
class ModelCard:
    name: str              # what you type after --model
    family: str            # "insectdct" | "flatbug" | "sam3" -> zoo/families/<family>.py
    title: str
    task: str              # "detection" or "instance segmentation"
    architecture: str
    weights: WeightFile
    default_threshold: float
    default_iou: float
    label: str             # class name written to the CSV
    min_ram_gb: float      # rough minimum free RAM to load + run on CPU
    min_vram_gb: float     # rough minimum free GPU memory to run on CUDA
    code_url: str
    paper: str
    doi: str
    license: str
    description: str
    extra_links: dict = field(default_factory=dict)
    text_prompt: bool = False  # True for models steered by text (e.g. SAM3: "bee"); unlocks the prompt box in the UI
    default_prompt: str = ""   # used when a text-prompt model gets no prompt
    gated: str = ""            # Hugging Face page where access must be requested (needs HF_TOKEN); "" = open download


# --------------------------------------------------------------------------- insectDCT (Bjerge et al.)
INSECTDCT_COMMIT = "e459ae87ebe39732963e77e0fcdd0a9a28672a08"     # pinned so the download is reproducible
_INSECTDCT_RAW = "https://raw.githubusercontent.com/kimbjerge/insectDCT/%s/runs/detect/" % INSECTDCT_COMMIT
_INSECTDCT = dict(
    family="insectdct",
    task="detection",
    default_iou=0.3,                     # upstream default NMS IoU
    label="insect",
    code_url="https://github.com/kimbjerge/insectDCT",
    paper="Bjerge, Wogram, Serra-Marin, Sakhiashvili & Høye (2026). InsectDCT: A generalized pipeline for detection, "
          "taxonomic classification, and tracking of insects in camera-trap recordings. bioRxiv.",
    doi="10.64898/2026.07.07.736939",
    license="GPL-3.0",
    extra_links={"Dataset (V6, Zenodo)": "https://zenodo.org/records/21154490"},
)

# --------------------------------------------------------------------------- flat-bug (Svenning et al.)
_FLATBUG_ERDA = "https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/"
_FLATBUG = dict(
    family="flatbug",
    task="instance segmentation",
    default_threshold=0.2,               # flat-bug DEFAULT_CFG
    default_iou=0.2,
    label="arthropod",
    code_url="https://github.com/darsa-group/flat-bug",
    paper="Svenning, Mougeot, Alison, Chevalier, Chavez Molina, Ong, Bjerge, Carrillo, Høye & Geissmann (2026). "
          "A general method for detection and segmentation of terrestrial arthropods in images. "
          "Methods in Ecology and Evolution 17(3), 727-739.",
    doi="10.1111/2041-210x.70249",
    license="MIT",
    extra_links={"Docs": "https://darsa.info/flat-bug/", "Dataset (Zenodo)": "https://doi.org/10.5281/zenodo.14761446"},
)

MODELS = {}


def _add(**kw):
    card = ModelCard(**kw)
    MODELS[card.name] = card


_add(name="insectdct-v8-m", title="insectDCT detector v8 (YOLO11m)", architecture="YOLO11m",
     weights=WeightFile("insects8Color.pt", (_INSECTDCT_RAW + "insects8Color/weights/best.pt",), 40730348,
                        "81f3ce9e89f2e3cf4ad6ba532f85c7c791f58e562875c8646419cfc2ee7de8e0"),
     default_threshold=0.407,            # upstream best-F1 confidence for insects8Color
     min_ram_gb=2, min_vram_gb=1,
     description="Insect detector for camera-trap images of flowers and vegetation, trained on dataset version 8. "
                 "The most accurate insectDCT colour detector.",
     **_INSECTDCT)
_add(name="insectdct-v8-s", title="insectDCT detector v8 (YOLO11s)", architecture="YOLO11s",
     weights=WeightFile("insects8Color11s.pt", (_INSECTDCT_RAW + "insects8Color11s/weights/best.pt",), 19392986,
                        "719ff88f1b8c9dc57ef0fed5e741adc12900a4c3e93c31f9a3c071379947e4ac"),
     default_threshold=0.407,            # no separate best-F1 value published for the 11s model; same as v8-m
     min_ram_gb=1.5, min_vram_gb=0.5,
     description="Smaller, faster version of insectdct-v8-m (made for edge devices such as a Raspberry Pi).",
     **_INSECTDCT)

for _size, _nbytes, _sha, _ram, _vram in [
        ("N", 6275129, "e54eb199fb8397f70dc36aca4727b92836fb7f56b11f8544be60f54c13d273c0", 2, 1),
        ("S", 21398649, "b410d2c902c8975f6c8263ee74871e618e55d670f86cf49ad49869de9a9ccc03", 2, 1.5),
        ("M", 49694305, "2ec28c5000b8d2bad38a702c96bfbda262a5f65dd643b0ba3a36b541bc3283aa", 3, 2),
        ("L", 84085321, "27ba525b4946221417a2f263e190cb44ada34923f3fdb678c79ac572ac743f79", 4, 3)]:
    _add(name="flatbug-" + _size.lower(), title="flat-bug %s (YOLOv8%s-seg)" % (_size, _size.lower()),
         architecture="YOLOv8%s-seg" % _size.lower(),
         weights=WeightFile("flat_bug_%s.pt" % _size, (_FLATBUG_ERDA + "flat_bug_%s.pt" % _size,), _nbytes, _sha),
         min_ram_gb=_ram, min_vram_gb=_vram,
         description="Detects and outlines (segments) all terrestrial arthropods, on any background and image size "
                     "(tiles large images in a pyramid). Tuned for top-down images and scans.",
         **_FLATBUG)
_add(name="flatbug-m-v2", title="flat-bug M v2 (YOLO26m-seg)", architecture="YOLO26m-seg",
     weights=WeightFile("flat_bug_M_v2.pt", (_FLATBUG_ERDA + "flat_bug_M_v2.pt",), 54585258,
                        "d3239bf0c367938f90dd21e92c4fb8bb47e28288678c5130c33f708714c8e50d"),
     min_ram_gb=3, min_vram_gb=2,
     description="Newest flat-bug model (default since flat-bug 1.2), built on YOLO26.",
     **_FLATBUG)


# --------------------------------------------------------------------------- SAM 3 (Meta), gated on Hugging Face
SAM3_COMMIT = "3c879f39826c281e95690f02c7821c4de09afae7"
_add(name="sam3", family="sam3", title="SAM 3 (Meta), finds what you describe", task="instance segmentation, text",
     architecture="SAM 3 (848M params)",
     weights=WeightFile("sam3.pt", ("https://huggingface.co/facebook/sam3/resolve/%s/sam3.pt" % SAM3_COMMIT,),
                        3450062241, "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"),
     default_threshold=0.5,               # SAM 3 processor default
     default_iou=0.5,
     label="",                            # the label is the text prompt
     min_ram_gb=10, min_vram_gb=6,
     code_url="https://github.com/facebookresearch/sam3",
     paper="Carion, Gustafson, Hu et al. (2025). SAM 3: Segment Anything with Concepts. arXiv:2511.16719.",
     doi="10.48550/arXiv.2511.16719",
     license="SAM License",
     description="Foundation model from Meta: type what to look for (e.g. 'bee', or 'bee, butterfly') and it outlines "
                 "every match. Not trained on insects specifically. Large (3.2 GB); a GPU is strongly recommended.",
     extra_links={"Hugging Face": "https://huggingface.co/facebook/sam3"},
     text_prompt=True, default_prompt="insect", gated="https://huggingface.co/facebook/sam3")


def get_model(name):
    """Return the ModelCard for `name` (case-insensitive) or raise KeyError with a helpful message."""
    key = (name or "").strip().lower()
    if key in MODELS:
        return MODELS[key]
    close = difflib.get_close_matches(key, MODELS.keys(), n=3, cutoff=0.4)
    msg = "Unknown model '%s'." % name
    if close:
        msg += " Did you mean: %s?" % ", ".join(close)
    raise KeyError(msg + "\n\n" + models_table())


def models_table():
    rows = [("MODEL", "TASK", "ARCHITECTURE", "WEIGHTS", "LICENSE", "ACCESS")]
    for c in MODELS.values():
        rows.append((c.name, c.task, c.architecture, size_text(c.weights.size), c.license,
                     "GATED *" if c.gated else "open"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    table = "\n".join("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in rows)
    if any(c.gated for c in MODELS.values()):
        table += "\n\n  * GATED = free, but you must request access and add a Hugging Face token first.\n" \
                 "    How to (5 minutes): " + GATED_GUIDE_URL
    return table


def size_text(nbytes):
    return "%.1f GB" % (nbytes / 1024 / MB) if nbytes >= 1024 * MB else "%.0f MB" % (nbytes / MB)
