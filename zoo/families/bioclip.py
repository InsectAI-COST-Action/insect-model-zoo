"""BioCLIP (Imageomics): zero-shot classification via open_clip - https://github.com/Imageomics/bioclip-2

Each insect is cropped from the image and compared with a text for every name; the best match wins.

- Without your own names BioCLIP picks from every taxon it knows: the species table of TreeOfLife-200M (text features
  pre-computed by the BioCLIP authors, CC0), limited to insects (about 250,000 species) unless BIOCLIP_INSECT_TAXA =
  False in main.py (then the whole tree of life, about 800,000 taxa). It answers at the deepest rank it is sure of:
  the species when that is clear, else the genus, family, order, ...
- With your own names (species, genera, orders, common names) it picks from those, compared together with the
  arthropod orders.
"""

import json
import os

import numpy as np

ARCHITECTURE = {"bioclip-2.5": "ViT-H-14", "bioclip-2": "ViT-L-14"}     # open_clip names of the two backbones
MARGIN = 0.1            # crop 10% around each box: a bit of context helps CLIP models
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


class Classifier:
    def __init__(self, card, weights_path, device):
        import open_clip
        self.card = card
        self.device = device
        self.status = lambda msg: None                              # set by the engine: the UI's status text
        self.folder = os.path.dirname(weights_path)
        arch = ARCHITECTURE[card.name]
        precision = "fp16" if device.startswith("cuda") else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=weights_path, device=device, precision=precision)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self._text = (None, None)                                   # (class list, text features) cache
        self._table = None                                          # species table, loaded on first use

    def _text_features(self, pairs):
        import torch
        if self._text[0] != pairs:
            with torch.inference_mode():
                feats = self.model.encode_text(self.tokenizer([p[1] for p in pairs]).to(self.device))
                self._text = (pairs, feats / feats.norm(dim=-1, keepdim=True))
        return self._text[1]

    def _species_table(self):
        from ..registry import BIOCLIP_INSECT_TAXA, display_name
        if self._table is None or self._table.insects_only != BIOCLIP_INSECT_TAXA:
            npy, names = (os.path.join(self.folder, f.filename) for f in self.card.species_table)
            self._table = None                                      # free the old one first
            cache = os.path.join(self.folder, "species_table_%s.npz" % ("insects" if BIOCLIP_INSECT_TAXA else "all"))
            self.status(("Loading %s's species list" if os.path.isfile(cache) else
                         "Preparing %s's species list (first use only, can take a minute)") % display_name(self.card))
            self._table = SpeciesTable(npy, names, BIOCLIP_INSECT_TAXA, self.device,
                                       self.model.visual.conv1.weight.dtype)
        return self._table

    def classify(self, image_rgb, detections, classes=None):
        """classes: your own names (list of str), compared together with the arthropod orders. None: pick from the
        species table (every insect species BioCLIP knows, or the whole tree of life)."""
        import torch
        from PIL import Image
        from ..registry import SPECIES_TABLE_SURE
        if not detections:
            return
        own = [c.strip() for c in (classes or []) if c and c.strip()]
        h, w = image_rgb.shape[:2]
        crops = []
        for d in detections:
            mx, my = (d.x2 - d.x1) * MARGIN, (d.y2 - d.y1) * MARGIN
            x1, y1 = max(0, int(d.x1 - mx)), max(0, int(d.y1 - my))
            x2, y2 = min(w, int(d.x2 + mx) + 1), min(h, int(d.y2 + my) + 1)
            crop = image_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((8, 8, 3), np.uint8)
            crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(crop))))

        if own or not self.card.species_table:
            text, pairs = self._own_names(own)
        else:
            table, text, pairs = self._species_table(), None, None
        dtype = self.model.visual.conv1.weight.dtype          # fp16 on GPU, fp32 on CPU
        scale = self.model.logit_scale.exp()
        for start in range(0, len(crops), 16):
            batch = torch.stack(crops[start:start + 16]).to(self.device, dtype)
            with torch.inference_mode():
                img = self.model.encode_image(batch)
                img = img / img.norm(dim=-1, keepdim=True)
                logits = scale * img @ (text.T if pairs else table.features)
                probs = logits.float().softmax(dim=-1).cpu().numpy()
            for d, p in zip(detections[start:start + 16], probs):
                if pairs:
                    best = int(p.argmax())
                    d.taxon, d.taxon_score, d.taxon_rank = pairs[best][0], float(p[best]), pairs[best][2]
                else:
                    d.taxon_options = table.options(p)             # species, genus, ..., kingdom (best of each)
                    fallback = d.taxon_options[-1] if d.taxon_options else ("Unsure", 0.0, "")
                    d.taxon, d.taxon_score, d.taxon_rank = next(
                        (o for o in d.taxon_options if o[1] >= SPECIES_TABLE_SURE), fallback)

    def _own_names(self, own):
        from ..registry import bioclip_default_classes
        # (name, text, rank). Your own names always compete with the arthropod orders (+ other life if
        # BIOCLIP_INSECT_TAXA = False): with only your names, one name would always score 1.00, and something that
        # fits none of them would still be forced into one of them.
        defaults = tuple(p for p in bioclip_default_classes() if p[0].lower() not in {c.lower() for c in own})
        pairs = tuple((c, "a photo of %s." % c, "") for c in own) + defaults
        return self._text_features(pairs), pairs


class SpeciesTable:
    """BioCLIP's text features for every taxon it knows, with the taxon's ranks.

    The downloaded table (float32, all ~800,000 taxa) is turned once into a smaller cache next to it: float16 and,
    with insects_only, only class Insecta. The cache also holds, for each rank, which group (e.g. which genus) every
    taxon belongs to, so the probability of a genus / family / order is the sum over its species."""

    def __init__(self, npy_path, names_path, insects_only, device, dtype):
        import torch
        self.insects_only = insects_only
        cache = os.path.join(os.path.dirname(npy_path), "species_table_%s" % ("insects" if insects_only else "all"))
        if not (os.path.isfile(cache + ".npz") and os.path.isfile(cache + ".json")):
            _build_cache(npy_path, names_path, insects_only, cache)
        with np.load(cache + ".npz") as data:
            self.groups = data["groups"]                            # (ranks, taxa): group number per rank
            features = data["features"]                             # (dim, taxa), float16
        with open(cache + ".json", encoding="utf-8") as f:
            self.names = json.load(f)                               # per rank: the name of each group
        self.features = torch.from_numpy(features).to(device, dtype)
        self.size = features.shape[1]

    def options(self, probs):
        """The best name at every rank with its probability, deepest (species) first: [(name, p, rank), ...]."""
        out = []
        for level in range(len(RANKS) - 1, -1, -1):
            summed = np.bincount(self.groups[level], weights=probs, minlength=len(self.names[level]))
            summed[0] = -1.0                                        # group 0 = rank not filled in: never an answer
            g = int(summed.argmax())
            if g:
                out.append((self.names[level][g], float(summed[g]), RANKS[level]))
        return out


def _build_cache(npy_path, names_path, insects_only, cache):
    print("Preparing BioCLIP's species table (%s, once) ..." % ("insects" if insects_only else "tree of life"))
    with open(names_path, encoding="utf-8") as f:
        rows = json.load(f)                                         # [[kingdom, ..., genus, species epithet], common]
    keep = [i for i, (ranks, _) in enumerate(rows) if not insects_only or ranks[2] == "Insecta"]
    table = np.load(npy_path, mmap_mode="r")                        # (dim, all taxa), float32
    features = np.empty((table.shape[0], len(keep)), np.float16)
    for r in range(0, table.shape[0], 32):                          # a few rows at a time: little memory
        features[r:r + 32] = table[r:r + 32][:, keep]
    groups = np.zeros((len(RANKS), len(keep)), np.int32)
    names = []
    for level in range(len(RANKS)):
        index, level_names = {}, [""]                               # group 0 = rank not filled in
        for j, i in enumerate(keep):
            ranks = rows[i][0]
            if level == len(RANKS) - 1:                             # species: "Genus epithet"
                name = "%s %s" % (ranks[5], ranks[6]) if ranks[5] and ranks[6] else ""
            else:
                name = ranks[level]
            if not name:
                continue
            key = (ranks[0], name)                                  # kingdom + name: a few names exist twice
            if key not in index:
                index[key] = len(level_names)
                level_names.append(name)
            groups[level, j] = index[key]
        names.append(level_names)
    with open(cache + ".npz.part", "wb") as f:
        np.savez(f, features=features, groups=groups)
    with open(cache + ".json.part", "w", encoding="utf-8") as f:
        json.dump(names, f)
    os.replace(cache + ".npz.part", cache + ".npz")
    os.replace(cache + ".json.part", cache + ".json")
