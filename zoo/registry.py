"""
The list of models in the zoo. One ModelCard per model; `family` picks the code in zoo/families/ that runs it.

Two kinds of models:
  - detectors   find the insects (boxes, sometimes outlines)            -> MODELS
  - classifiers say what each found insect is (taxon + score)          -> CLASSIFIERS
A detector can name a default classifier to pair with (insectDCT detector -> insectDCT classifier); any detector can be
combined with any classifier.

To add a model: add a ModelCard below (and a family module if it is a new kind of model), then add a row to the
"Models" table in README.md.
"""

import difflib
from dataclasses import dataclass, field

MB = 1024 * 1024
REPO_URL = "https://github.com/InsectAI-COST-Action/insect-model-zoo"
GATED_GUIDE_URL = REPO_URL + "/blob/main/docs/GATED_MODELS.md"    # how to get access + where to put the token


@dataclass(frozen=True)
class WeightFile:
    filename: str
    urls: tuple            # tried in order, so a mirror (e.g. Hugging Face) can be added in front later
    size: int              # bytes; used for the progress bar and the free-disk check
    sha256: str            # checked after download


@dataclass(frozen=True)
class ModelCard:
    name: str              # what you type after --model / --classifier
    family: str            # zoo/families/<family>.py
    title: str
    task: str
    architecture: str
    weights: WeightFile
    default_threshold: float
    default_iou: float
    label: str             # class name written to the CSV (detectors)
    min_ram_gb: float      # rough minimum free RAM to load + run on CPU
    min_vram_gb: float     # rough minimum free GPU memory to run on CUDA
    code_url: str
    paper: str
    doi: str
    license: str
    description: str
    extra_links: dict = field(default_factory=dict)
    kind: str = "detector"     # "detector" or "classifier"
    default_classifier: str = ""   # detectors: classifier used with it by default ("" = none)
    text_prompt: bool = False  # True for models steered by text (e.g. SAM3: "bee"); unlocks the prompt box in the UI
    default_prompt: str = ""   # used when a text-prompt model gets no prompt
    classes: tuple = ()        # zero-shot classifiers: default (name, text) pairs; you can give your own names instead
    extra_files: tuple = ()    # (relative path, url) of small extra files the model needs (e.g. upstream code)
    group: str = ""            # model family shown once in the UI list, e.g. "flat-bug" ("" = the name itself)
    variant: str = ""          # size / version within the family, e.g. "M" (shown when the family is picked)
    gated: str = ""            # Hugging Face page where access must be requested (needs HF_TOKEN); "" = open download
    species_table: tuple = ()  # zero-shot classifiers: WeightFiles of pre-computed text features for every taxon the
                               # model knows (used when you give no names); downloaded next to the weights


MODELS = {}         # detectors
CLASSIFIERS = {}


def _add(**kw):
    card = ModelCard(**kw)
    (CLASSIFIERS if card.kind == "classifier" else MODELS)[card.name] = card


# =========================================================================== DETECTORS
# --------------------------------------------------------------------------- insectDCT (Bjerge et al.)
INSECTDCT_COMMIT = "e459ae87ebe39732963e77e0fcdd0a9a28672a08"     # pinned so the download is reproducible
_INSECTDCT_RAW = "https://raw.githubusercontent.com/kimbjerge/insectDCT/%s/" % INSECTDCT_COMMIT
_INSECTDCT_PAPER = dict(
    code_url="https://github.com/kimbjerge/insectDCT",
    paper="Bjerge, Wogram, Serra-Marin, Sakhiashvili & Høye (2026). InsectDCT: A generalized pipeline for detection, "
          "taxonomic classification, and tracking of insects in camera-trap recordings. bioRxiv.",
    doi="10.64898/2026.07.07.736939",
    license="GPL-3.0",
    extra_links={"Dataset (V6, Zenodo)": "https://zenodo.org/records/21154490"},
)
_INSECTDCT = dict(family="insectdct", task="detection", default_iou=0.3, label="insect", group="insectDCT v8",
                  default_classifier="insectdct-cls-v7", **_INSECTDCT_PAPER)

_add(name="insectdct-v8-m", title="insectDCT detector v8 (YOLO11m)", architecture="YOLO11m", variant="M",
     weights=WeightFile("insects8Color.pt", (_INSECTDCT_RAW + "runs/detect/insects8Color/weights/best.pt",), 40730348,
                        "81f3ce9e89f2e3cf4ad6ba532f85c7c791f58e562875c8646419cfc2ee7de8e0"),
     default_threshold=0.407,            # upstream best-F1 confidence for insects8Color
     min_ram_gb=2, min_vram_gb=1,
     description="Insect detector for camera-trap images of flowers and vegetation, trained on dataset version 8. "
                 "The most accurate insectDCT colour detector.",
     **_INSECTDCT)
_add(name="insectdct-v8-s", title="insectDCT detector v8 (YOLO11s)", architecture="YOLO11s", variant="S",
     weights=WeightFile("insects8Color11s.pt", (_INSECTDCT_RAW + "runs/detect/insects8Color11s/weights/best.pt",),
                        19392986, "719ff88f1b8c9dc57ef0fed5e741adc12900a4c3e93c31f9a3c071379947e4ac"),
     default_threshold=0.407,            # no separate best-F1 value published for the 11s model; same as v8-m
     min_ram_gb=1.5, min_vram_gb=0.5,
     description="Smaller, faster version of insectdct-v8-m (made for edge devices such as a Raspberry Pi).",
     **_INSECTDCT)

# --------------------------------------------------------------------------- flat-bug (Svenning et al.)
_FLATBUG_ERDA = "https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/"
_FLATBUG = dict(
    family="flatbug",
    group="flat-bug",
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

for _size, _nbytes, _sha, _ram, _vram in [
        ("N", 6275129, "e54eb199fb8397f70dc36aca4727b92836fb7f56b11f8544be60f54c13d273c0", 2, 1),
        ("S", 21398649, "b410d2c902c8975f6c8263ee74871e618e55d670f86cf49ad49869de9a9ccc03", 2, 1.5),
        ("M", 49694305, "2ec28c5000b8d2bad38a702c96bfbda262a5f65dd643b0ba3a36b541bc3283aa", 3, 2),
        ("L", 84085321, "27ba525b4946221417a2f263e190cb44ada34923f3fdb678c79ac572ac743f79", 4, 3)]:
    _add(name="flatbug-" + _size.lower(), title="flat-bug %s (YOLOv8%s-seg)" % (_size, _size.lower()),
         architecture="YOLOv8%s-seg" % _size.lower(), variant=_size,
         weights=WeightFile("flat_bug_%s.pt" % _size, (_FLATBUG_ERDA + "flat_bug_%s.pt" % _size,), _nbytes, _sha),
         min_ram_gb=_ram, min_vram_gb=_vram,
         description="Detects and outlines (segments) all terrestrial arthropods, on any background and image size "
                     "(tiles large images in a pyramid). Tuned for top-down images and scans.",
         **_FLATBUG)
_add(name="flatbug-m-v2", title="flat-bug M v2 (YOLO26m-seg)", architecture="YOLO26m-seg", variant="M v2",
     weights=WeightFile("flat_bug_M_v2.pt", (_FLATBUG_ERDA + "flat_bug_M_v2.pt",), 54585258,
                        "d3239bf0c367938f90dd21e92c4fb8bb47e28288678c5130c33f708714c8e50d"),
     min_ram_gb=3, min_vram_gb=2,
     description="Newest flat-bug model (default since flat-bug 1.2), built on YOLO26.",
     **_FLATBUG)

# --------------------------------------------------------------------------- SAM 3 (Meta), gated on Hugging Face
SAM3_COMMIT = "3c879f39826c281e95690f02c7821c4de09afae7"
_add(name="sam3", family="sam3", group="SAM 3", title="SAM 3 (Meta), finds what you describe",
     task="instance segmentation, text",
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

# =========================================================================== CLASSIFIERS
# --------------------------------------------------------------------------- insectDCT hierarchical classifier
_add(name="insectdct-cls-v7", kind="classifier", family="insectdct_cls", group="insectDCT V7", variant="V7",
     title="insectDCT hierarchical classifier V7 (ConvNeXt-Base)",
     task="104 insect taxa (order / family / species)", architecture="ConvNeXt-Base, 224 px crops",
     weights=WeightFile("HierarchicalClassifierV7.zip",
                        ("https://drive.usercontent.google.com/download?id=15oGWBgp3S08k8VK0r2qzMC65uUBFvsPh"
                         "&export=download&confirm=t",),        # upstream README's Google Drive link (V7)
                        507057210, "da89fdfc64014c1d8889a19bd6baa80eac409600f290ce671327cf1fa19425ad"),
     default_threshold=0.0, default_iou=0.0, label="",
     min_ram_gb=3, min_vram_gb=1.5,
     description="Classifies each detected insect to species, genus, family or order: it goes as deep as it is sure "
                 "(per-class thresholds), otherwise says 'Unsure'. Trained on camera-trap crops of flower visitors.",
     extra_files=tuple(("upstream/common/" + f, _INSECTDCT_RAW + "common/" + f) for f in
                       ("__init__.py", "hierarchical_classifier.py", "resnet50tf.py", "convNext.py", "efficientNet.py")),
     **_INSECTDCT_PAPER)

# --------------------------------------------------------------------------- BioCLIP (Imageomics), zero-shot
# Default names: arthropod orders, written as BioCLIP's taxonomic text (kingdom ... order), which it was trained on.
_ORDERS = [("Hymenoptera", "Insecta"), ("Diptera", "Insecta"), ("Coleoptera", "Insecta"), ("Lepidoptera", "Insecta"),
           ("Hemiptera", "Insecta"), ("Orthoptera", "Insecta"), ("Odonata", "Insecta"), ("Neuroptera", "Insecta"),
           ("Trichoptera", "Insecta"), ("Ephemeroptera", "Insecta"), ("Dermaptera", "Insecta"),
           ("Blattodea", "Insecta"), ("Mantodea", "Insecta"), ("Araneae", "Arachnida"), ("Opiliones", "Arachnida"),
           ("Isopoda", "Malacostraca")]
ARTHROPOD_ORDERS = tuple((order, "a photo of Animalia Arthropoda %s %s." % (cls, order), "order")
                         for order, cls in _ORDERS)
# Added when BIOCLIP_INSECT_TAXA is False, so things that are not insects (e.g. a flower found by the detector) can
# be named as such instead of being forced into an insect order. (name, BioCLIP text, rank)
OTHER_LIFE = (("plant", "a photo of Plantae Tracheophyta Magnoliopsida.", "class"),
              ("fungus", "a photo of Fungi.", "kingdom"),
              ("bird", "a photo of Animalia Chordata Aves.", "class"),
              ("mammal", "a photo of Animalia Chordata Mammalia.", "class"),
              ("reptile", "a photo of Animalia Chordata Reptilia.", "class"),
              ("amphibian", "a photo of Animalia Chordata Amphibia.", "class"),
              ("snail or slug", "a photo of Animalia Mollusca Gastropoda.", "class"),
              ("earthworm", "a photo of Animalia Annelida Clitellata.", "class"))
BIOCLIP_INSECT_TAXA = True     # set from main.py: True = BioCLIP picks from insects only; False = whole tree of life
SPECIES_TABLE_SURE = 0.5       # BioCLIP without names answers at the deepest rank at least this sure (UI: the slider)


def bioclip_default_classes():
    """The names your own BioCLIP names compete with: arthropod orders, plus other life if BIOCLIP_INSECT_TAXA is
    False. (Without your own names BioCLIP picks from its whole species table instead, see zoo/families/bioclip.py.)"""
    return ARTHROPOD_ORDERS if BIOCLIP_INSECT_TAXA else ARTHROPOD_ORDERS + OTHER_LIFE


_LATIN = None


def clean_latin_names(names):
    """BioCLIP names as Latin (scientific) names, whatever the upper/lower case: 'apis' -> 'Apis',
    'bombus TERRESTRIS' -> 'Bombus terrestris'. A genus, family or order alone is fine too.
    Returns (cleaned names, problems); problems lists names that cannot be Latin names (digits, symbols, ...)."""
    global _LATIN
    import re
    if _LATIN is None:
        _LATIN = re.compile(r"^[A-Z][a-z]+( [a-z]+(-[a-z]+)?){0,2}$")
    cleaned, problems = [], []
    for name in names:
        words = name.split()
        fixed = " ".join([words[0].capitalize()] + [w.lower() for w in words[1:]]) if words else ""
        if _LATIN.match(fixed):
            cleaned.append(fixed)
        else:
            problems.append("'%s' is not a Latin name." % name)
    if problems:
        problems.append("Please write the complete Latin name as: Genus species, e.g. Apis mellifera or Bombus "
                        "terrestris. A genus, family or order alone also works, e.g. Bombus, Syrphidae, Hymenoptera.")
    return cleaned, problems


def bioclip_default_text():
    """What your own BioCLIP names are always compared with."""
    return "arthropod orders" if BIOCLIP_INSECT_TAXA else "arthropod orders + plants, fungi, birds, mammals, ..."


def bioclip_empty_text():
    """What BioCLIP picks from when you give no names."""
    return ("every insect species it knows (about 250,000)" if BIOCLIP_INSECT_TAXA
            else "every taxon in the tree of life it knows (about 800,000)")


TOL_REVISION = "5f2dc493b3dc0e544438a04038ab15faa646b749"      # imageomics/TreeOfLife-200M (dataset, CC0), pinned
_TOL = "https://huggingface.co/datasets/imageomics/TreeOfLife-200M/resolve/%s/embeddings/" % TOL_REVISION


def _tol(stem, npy_size, npy_sha, json_size, json_sha):
    return (WeightFile(stem + ".npy", (_TOL + stem + ".npy",), npy_size, npy_sha),
            WeightFile(stem + ".json", (_TOL + stem + ".json",), json_size, json_sha))


_BIOCLIP = dict(
    kind="classifier", family="bioclip", default_threshold=0.0, default_iou=0.0, label="",
    code_url="https://github.com/Imageomics/bioclip-2",
    paper="Gu, Stevens, Campolongo et al. (2025). BioCLIP 2: Emergent Properties from Scaling Hierarchical "
          "Contrastive Learning. arXiv:2505.23883.",
    doi="10.48550/arXiv.2505.23883",
    license="MIT",
    classes=ARTHROPOD_ORDERS,
)
_add(name="bioclip-2.5", title="BioCLIP 2.5 Huge (zero-shot, any names)", architecture="ViT-H/14 (open_clip)",
     group="BioCLIP 2.5", variant="Huge",
     task="any names you give (zero-shot); none given: ~250,000 insect species (or the whole tree of life)",
     weights=WeightFile("open_clip_model.safetensors",
                        ("https://huggingface.co/imageomics/bioclip-2.5-vith14/resolve/"
                         "6e3d04e3d6522012c88181085c5ae666e14c45cd/open_clip_model.safetensors",),
                        3944517804, "ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97"),
     species_table=_tol("txt_emb_bioclip-2.5-vith14",
                        3255820416, "d1cc734330d17ea26e6f713b289b2138b4adc90d4f524349cd16fab42d5c3358",
                        84142188, "af0cb41ffbfb31e6a2e2d5e3a402529ec8245a4268f42ce45ee4e977b7127443"),
     min_ram_gb=9, min_vram_gb=3.5,
     description="Biology foundation model trained on 200M+ images of the tree of life. Without names it picks from "
                 "every insect species it knows; or give it names (species, genera, orders). The strongest BioCLIP.",
     extra_links={"Hugging Face": "https://huggingface.co/imageomics/bioclip-2.5-vith14"},
     **_BIOCLIP)
_add(name="bioclip-2", title="BioCLIP 2 (zero-shot, any names)", architecture="ViT-L/14 (open_clip)", group="BioCLIP 2", variant="L",
     task="any names you give (zero-shot); none given: ~260,000 insect species (or the whole tree of life)",
     weights=WeightFile("open_clip_model.safetensors",
                        ("https://huggingface.co/imageomics/bioclip-2/resolve/"
                         "2957b322090f9cb17ae72c71981c7218a28d81e0/open_clip_model.safetensors",),
                        1710517724, "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"),
     species_table=_tol("txt_emb_bioclip-2",
                        2664821888, "c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2",
                        91586174, "4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5"),
     min_ram_gb=5, min_vram_gb=2.5,
     description="Smaller, faster BioCLIP (half the size of 2.5) with the same way of working.",
     extra_links={"Hugging Face": "https://huggingface.co/imageomics/bioclip-2"},
     **_BIOCLIP)


def tags(card):
    """Short labels shown as coloured tags in the UI and in the model lists."""
    if card.kind == "classifier":
        out = ["classifier", "zero-shot" if card.classes else "hierarchical"]
    else:
        out = ["detector"] + (["segmentation"] if "segmentation" in card.task else [])
    if card.text_prompt:
        out.append("text prompt")
    if card.gated:
        out.append("gated")
    return out


def groups(table):
    """{family: [cards, ...]} in registry order; a model without a family is its own family."""
    out = {}
    for card in table.values():
        out.setdefault(card.group or card.name, []).append(card)
    return out


def group_of(card):
    return card.group or card.name


def display_name(card):
    """Name for people: 'SAM 3', 'flat-bug M v2', 'BioCLIP 2.5' (card.name is the short code you type, e.g. 'sam3')."""
    group = group_of(card)
    sizes = sum(1 for table in (MODELS, CLASSIFIERS) for c in table.values() if group_of(c) == group)
    return "%s %s" % (group, card.variant) if card.variant and sizes > 1 else group


def group_default(group, table):
    """The version picked when a family is chosen in the UI: the largest one (usually the most accurate)."""
    return max(groups(table)[group], key=lambda card: card.weights.size)


def _lookup(name, table, what):
    key = (name or "").strip().lower()
    if key in table:
        return table[key]
    close = difflib.get_close_matches(key, table.keys(), n=3, cutoff=0.4)
    msg = "Unknown %s '%s'." % (what, name)
    if close:
        msg += " Did you mean: %s?" % ", ".join(close)
    raise KeyError(msg + "\n\n" + models_table())


def get_model(name):
    """Return the detector ModelCard for `name` (case-insensitive) or raise KeyError with a helpful message."""
    return _lookup(name, MODELS, "model")


def get_classifier(name, detector=None):
    """'auto' = the detector's own classifier (or none), 'none'/'' = no classifier, else a classifier name."""
    key = (name or "").strip().lower()
    if key == "auto":
        key = detector.default_classifier if detector is not None else ""
    if key in ("", "none"):
        return None
    return _lookup(key, CLASSIFIERS, "classifier")


def models_table():
    def table(rows):
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return "\n".join("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in rows)

    access = lambda c: "GATED *" if c.gated else "open"                                   # noqa: E731
    tag_text = lambda c: ", ".join(t for t in tags(c) if t != "gated")                   # noqa: E731
    det = [("DETECTOR (-m)", "TAGS", "ARCHITECTURE", "WEIGHTS", "LICENSE", "ACCESS", "DEFAULT CLASSIFIER")]
    det += [(c.name, tag_text(c), c.architecture, size_text(download_size(c)), c.license, access(c),
             c.default_classifier or "-") for c in MODELS.values()]
    cls = [("CLASSIFIER (-c)", "TAGS", "CLASSES", "ARCHITECTURE", "WEIGHTS", "LICENSE", "ACCESS")]
    cls += [(c.name, tag_text(c), c.task, c.architecture, size_text(download_size(c)), c.license, access(c))
            for c in CLASSIFIERS.values()]
    text = table(det) + "\n\n" + table(cls)
    if any(c.gated for c in list(MODELS.values()) + list(CLASSIFIERS.values())):
        text += "\n\n  * GATED = free, but you must request access and add a Hugging Face token first.\n" \
                "    How to (5 minutes): " + GATED_GUIDE_URL
    return text


def download_size(card):
    """Bytes downloaded for a model: its weights plus (zero-shot classifiers) its species table."""
    return card.weights.size + sum(f.size for f in card.species_table)


def size_text(nbytes):
    return "%.1f GB" % (nbytes / 1024 / MB) if nbytes >= 1024 * MB else "%.0f MB" % (nbytes / MB)
