"""insectDCT hierarchical classifier (ConvNeXt-Base, V7) - https://github.com/kimbjerge/insectDCT

Runs the upstream classifier code (GPL-3.0), which is downloaded next to the weights at a pinned commit
(see extra_files in zoo/registry.py). Each insect is cropped the same way as upstream and classified to
order -> family -> genus/species, as deep as the per-class thresholds say it is sure; otherwise "Unsure".
"""

import contextlib
import glob
import io
import os
import pickle
import sys
import zipfile

import numpy as np

TAG, MODEL_NAME, CROP = "CNB", "ConvNextBase", 224     # V7 ConvNeXt-Base, trained on 224 x 224 crops

# For Camtrap DP (scientific names only): the classes that are not plain Latin names -> (name, comment).
# None = not an animal (left out of the observations).
_SCIENTIFIC = {"Aranaea": ("Araneae", ""), "Birds": ("Aves", ""), "Formidicidae": ("Formicidae", ""),
               "Hesperidae": ("Hesperiidae", ""), "Milipedes": ("Diplopoda", ""), "Moths": ("Lepidoptera", "moth"),
               "Slugs": ("Gastropoda", "slug"), "Snails": ("Gastropoda", "snail"),
               "Fritillaries": ("Nymphalidae", "fritillary"), "Larvae": ("Insecta", "larva"),
               "Herpetofauna": ("Chordata", "reptile or amphibian"), "Hymenoptera_bees": ("Hymenoptera", "bee"),
               "Hymenoptera_nobees": ("Hymenoptera", "not a bee"),
               "Sphaerophoria scripta-complex": ("Sphaerophoria", "Sphaerophoria scripta complex"),
               "Vegetation": None}
_ORDERS = {"Araneae", "Coleoptera", "Dermaptera", "Diptera", "Hemiptera", "Hymenoptera", "Isopoda", "Lepidoptera",
           "Odonata", "Orthoptera"}
_CLASSES = {"Aves", "Insecta", "Diplopoda", "Gastropoda"}


def scientific_name(taxon):
    """insectDCT class name -> (scientific name, rank, comment) for Camtrap DP; None = not an animal.
    E.g. 'Aranaea' -> Araneae (order), 'Aglais urticae_fw' -> Aglais urticae (species), 'Apoidea small' -> Apoidea."""
    own = taxon
    if taxon in _SCIENTIFIC:
        if _SCIENTIFIC[taxon] is None:
            return None
        taxon, comment = _SCIENTIFIC[taxon]
    else:
        comment = ""
        if taxon.endswith("_fw"):
            taxon = taxon[:-3]
        if taxon.startswith("Apoidea "):                           # 'Apoidea small', 'Apoidea red_abdomen', ...
            taxon, comment = "Apoidea", taxon[len("Apoidea "):].replace("_", " ")
    if own != taxon:
        comment = "insectDCT class: %s%s" % (own, "; " + comment if comment else "")
    rank = _rank(taxon)
    return taxon, "" if rank in ("subfamily", "superfamily") else rank, comment   # Camtrap DP has no sub/super ranks


def taxon_rank(taxon):
    """Rank of an insectDCT class name, for the CSV / COCO output: 'Apis mellifera' -> species, 'Bombus' -> genus,
    'Apoidea small' -> superfamily, 'Birds' -> class; '' for Vegetation."""
    mapped = scientific_name(taxon)
    return "" if mapped is None else _rank(mapped[0])


def _rank(name):
    if len(name.split()) == 2:
        return "species"
    if name in _ORDERS:
        return "order"
    if name in _CLASSES:
        return "class"
    if name == "Chordata":
        return "phylum"
    if name.endswith("idae"):
        return "family"
    if name.endswith("inae"):
        return "subfamily"
    if name.endswith("oidea"):
        return "superfamily"
    return "genus"


class Classifier:
    def __init__(self, card, weights_path, device):
        folder = os.path.dirname(weights_path)
        files = _unpack(weights_path, folder)
        upstream = os.path.join(folder, "upstream")
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        import torchvision.models as tvm
        import common.convNext as convnext_module
        # upstream builds ConvNeXt with ImageNet weights (a 350 MB download) that the classifier weights replace anyway
        convnext_module.convnext_base = lambda weights=None: tvm.convnext_base(weights=None)
        from common.hierarchical_classifier import HierarchicalClassifier

        with open(files["labels"], "rb") as f:
            _, h1, h2, l1, l2, l3, *_ = pickle.load(f)
        with contextlib.redirect_stdout(io.StringIO()):                      # upstream prints a lot while loading
            self.clf = HierarchicalClassifier(h1, h2, l1, l2, l3, img_size=CROP, stdThreshold=0.0, device=device)
            self.clf.loadmodel(files["weights"], files["thresholds"], modelName=MODEL_NAME)
        self.n_taxa = len(l3)

    def classify(self, image_rgb, detections, classes=None):          # fixed classes; `classes` is not used
        import torch
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])               # upstream expects OpenCV (BGR) crops
        h, w = bgr.shape[:2]
        for d in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in (d.x1, d.y1, d.x2, d.y2))
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            half = (max(x2 - x1, y2 - y1) + 3) // 2                         # square crop, as upstream classifyInsect()
            crop = bgr[max(0, yc - half):min(h, yc + half), max(0, xc - half):min(w, xc + half)]
            if crop.size == 0:
                continue
            with torch.inference_mode():
                # upstream decides the taxon and how deep it is sure (order / family / species) ...
                _line, level, index, _name, _tail = self.clf.makePrediction(np.ascontiguousarray(crop),
                                                                             strongCheck=False)
                if index < 0:
                    d.taxon, d.taxon_score, d.taxon_rank = "Unsure", None, ""
                    continue
                # ... and the score we report is the plain softmax probability of that taxon at that level
                outputs = self.clf.model(self.clf.imagesInBatch.to(self.clf.device))
                d.taxon_score = float(outputs[level - 1].float().softmax(-1)[0, index])
            d.taxon = _name
            d.taxon_rank = taxon_rank(_name)        # a real rank: upstream level 3 mixes genus and species


def _unpack(zip_path, folder):
    def find(pattern):
        hits = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
        return hits[0] if hits else None

    names = {"weights": "HierarchicalClassifier_%s_V7.pth" % TAG, "labels": "HierarchicalLabels3L_%s_V7.pkl" % TAG,
             "thresholds": "HierarchicalThresholds3S_%s_V7.csv" % TAG}
    if not all(find(n) for n in names.values()):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.join(folder, "unpacked"))
    files = {k: find(n) for k, n in names.items()}
    missing = [n for k, n in names.items() if not files[k]]
    if missing:
        raise RuntimeError("Classifier files missing after unpacking %s: %s" % (zip_path, missing))
    return files
