"""Load models from the zoo (download weights -> hardware check -> load) and run detector -> classifier.
Shared by the CLI and the UI."""

import gc
import importlib
import time

from . import hardware, results
from .registry import display_name
from .weights import ensure_weights


class _Loaded:
    """One loaded model (detector or classifier) and the device it runs on."""

    def __init__(self, kind):
        self.kind = kind                    # "detector" or "classifier"
        self.card = self.model = self.device = None


class Zoo:
    """Keeps one detector and one classifier loaded, so switching models in the UI does not fill up GPU memory."""

    def __init__(self, device="auto", log=print, status=None):
        self.hw = hardware.probe()
        self.requested_device = device
        self.log = log
        self.status = status or (lambda msg: None)      # short "what is happening now" text (the UI's button)
        self.detector = _Loaded("detector")
        self.classifier = _Loaded("classifier")
        self.cpu_only = set()       # models that failed on the GPU: straight to the CPU until the app restarts

    # the detector is "the model" for code that only cares about detection
    card = property(lambda self: self.detector.card)
    device = property(lambda self: self.detector.device)

    def load(self, card, progress=None):
        """Load a detector. progress(done, total, message) is forwarded to the weight download."""
        return self._load(self.detector, card, progress)

    def load_classifier(self, card, progress=None):
        """Load a classifier, or unload it with card=None."""
        if card is None:
            self._unload(self.classifier)
            return None
        return self._load(self.classifier, card, progress)

    def _load(self, slot, card, progress=None, device=None):
        seen = len(self.hw.notes)
        if slot.card is not None and slot.card is not card:
            self._unload(slot)              # the model this one replaces goes first, so the memory check below sees
                                            # what is really free (not: BioCLIP 2 still loaded -> BioCLIP 2.5 to CPU)
        hardware.refresh(self.hw)                                   # free GPU memory changes as models load
        device = device or hardware.pick_device(self.requested_device, self.hw)
        if card.name in self.cpu_only:
            device = "cpu"
        device, warnings = hardware.check_model(card, device, self.hw)
        for w in self.hw.notes[seen:] + warnings:
            self.log("WARNING: " + w)
        if slot.card is card and slot.device == device:
            return slot.model

        path = ensure_weights(card, progress)
        self._unload(slot)
        self.status("Loading %s on %s" % (display_name(card), _where(device)))
        self.log("Loading %s %s on %s" % (slot.kind, display_name(card), hardware.describe_device(device, self.hw)))
        family = importlib.import_module("zoo.families." + card.family)
        build = family.Classifier if slot.kind == "classifier" else family.Model
        try:
            slot.model = build(card, path, device)
        except Exception as e:
            if device == "cpu":
                raise
            sticky = _is_gpu_error(e)               # a damaged file, say, is not the GPU's fault: next time, GPU again
            self.log("WARNING: %s could not be loaded on the GPU (%s), so it runs on the CPU instead (slower)%s."
                     % (display_name(card), _short(e), " until the app is restarted" if sticky else ""))
            self.status("%s did not load on the GPU, loading it on the CPU instead" % display_name(card))
            if sticky:
                self.cpu_only.add(card.name)
            self._unload(slot)                      # free what the failed GPU attempt took
            device = "cpu"
            slot.model = build(card, path, device)
        slot.card, slot.device = card, device
        if hasattr(slot.model, "status"):          # models with slow first-use steps say so (e.g. BioCLIP's table)
            slot.model.status = lambda msg: self.status(msg)
        return slot.model

    def _unload(self, slot):
        slot.model = slot.card = slot.device = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self):
        self._unload(self.detector)
        self._unload(self.classifier)

    def _on_cpu_if_gpu_fails(self, slot, run):
        """Run `run(model)`; if the GPU runs out of memory (or fails), reload this model on CPU and retry once."""
        try:
            return run(slot.model)
        except Exception as e:
            if slot.device == "cpu" or not _is_gpu_error(e):
                raise
            card = slot.card
            self.log("WARNING: %s failed on the GPU (%s), so it runs on the CPU instead (slower) until the app "
                     "is restarted." % (display_name(card), _short(e)))
            self.cpu_only.add(card.name)
            self.status("%s failed on the GPU, retrying on the CPU" % display_name(card))
            self._unload(slot)
            self._load(slot, card, device="cpu")
            return run(slot.model)

    def predict(self, image_rgb, threshold, iou, prompt=None):
        """Detect. `prompt` is the text prompt for models that take one (card.text_prompt), ignored otherwise."""
        prompt = ((prompt or "").strip() or None) if self.card.text_prompt else None
        self.status("Finding insects with %s on %s" % (display_name(self.card), _where(self.device)))
        t = time.time()
        dets = self._on_cpu_if_gpu_fails(self.detector, lambda m: m.predict(image_rgb, threshold, iou, prompt))
        h, w = image_rgb.shape[:2]
        for d in dets:                                              # keep boxes (and outlines) inside the image
            d.x1, d.x2 = (min(max(v, 0), w - 1) for v in (d.x1, d.x2))
            d.y1, d.y2 = (min(max(v, 0), h - 1) for v in (d.y1, d.y2))
            if d.polygon is not None:
                d.polygon = [[min(max(float(x), 0.0), w - 1.0), min(max(float(y), 0.0), h - 1.0)]
                             for x, y in d.polygon]
        return dets, time.time() - t

    def classify(self, image_rgb, dets, classes=None):
        """Give every detection a taxon with the loaded classifier (if any). Returns the seconds it took."""
        if self.classifier.model is None or not dets:
            return 0.0
        t = time.time()
        self._on_cpu_if_gpu_fails(self.classifier, lambda m: m.classify(image_rgb, dets, classes))
        return time.time() - t

    def run_file(self, path, threshold, iou, out_dir, prompt=None, classes=None, cls_threshold=None):
        image = results.load_image(path)
        dets, secs = self.predict(image, threshold, iou, prompt)
        if self.classifier.model is not None and dets:
            self.status("Naming %d insect%s with %s" % (len(dets), "" if len(dets) == 1 else "s",
                                                       display_name(self.classifier.card)))
        secs += self.classify(image, dets, classes)
        _apply_taxon_threshold(dets, cls_threshold)
        self.status("Saving results")
        meta = {"config": {"threshold": threshold, "iou": iou, "prompt": prompt, "classes": classes,
                           "cls_threshold": cls_threshold}, "device": self.device}
        files = results.save(path, image, dets, out_dir, self.card.name,
                             self.classifier.card.name if self.classifier.card else "", meta)
        return image, dets, secs, files

    def run_classify_only(self, path, out_dir, classes=None, cls_threshold=None):
        """No detector: treat the whole image as one box and just classify it."""
        image = results.load_image(path)
        h, w = image.shape[:2]
        dets = [results.Detection(0.0, 0.0, float(w - 1), float(h - 1), None, "")]   # no detector: no score
        self.status("Classifying the whole image with %s" % display_name(self.classifier.card))
        secs = self.classify(image, dets, classes)
        _apply_taxon_threshold(dets, cls_threshold)
        name = self.classifier.card.name if self.classifier.card else ""
        self.status("Saving results")
        meta = {"config": {"classes": classes, "cls_threshold": cls_threshold}, "device": self.classifier.device}
        files = results.save(path, image, dets, out_dir, "whole-image", name, meta)
        return image, dets, secs, files


def _apply_taxon_threshold(dets, cls_threshold):
    """The classification-confidence threshold (None = leave the classifier's own answer).
    With a name per rank (BioCLIP's species table): answer at the deepest rank at least this sure, so lower = more
    specific (species), higher = surer (genus, family, order). Other classifiers: a name below it becomes 'Unsure'."""
    if cls_threshold is None:
        return
    for d in dets:
        if d.taxon_options:
            d.taxon, d.taxon_score, d.taxon_rank = next(
                (o for o in d.taxon_options if o[1] >= cls_threshold), ("Unsure", d.taxon_options[-1][1], ""))
        elif d.taxon and d.taxon != "Unsure" and (d.taxon_score or 0.0) < cls_threshold:
            d.taxon = "Unsure"


def _is_gpu_error(e):
    text = "%s %s" % (type(e).__name__, e)
    return any(k in text for k in ("OutOfMemory", "out of memory", "CUDA", "cuda", "MPS", "mps", "cuDNN"))


def _where(device):
    """'cuda:0' -> 'GPU', 'mps' -> 'Apple GPU', 'cpu' -> 'CPU' (for the short status text)."""
    return "GPU" if device.startswith("cuda") else "Apple GPU" if device == "mps" else "CPU"


def _short(e):
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
