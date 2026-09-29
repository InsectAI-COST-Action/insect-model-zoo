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

    def __init__(self, device="auto", log=print):
        self.hw = hardware.probe()
        self.requested_device = device
        self.log = log
        self.detector = _Loaded("detector")
        self.classifier = _Loaded("classifier")

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
        hardware.refresh(self.hw)                                   # free GPU memory changes as models load
        device = device or hardware.pick_device(self.requested_device, self.hw)
        device, warnings = hardware.check_model(card, device, self.hw)
        for w in self.hw.notes[seen:] + warnings:
            self.log("WARNING: " + w)
        if slot.card is card and slot.device == device:
            return slot.model

        path = ensure_weights(card, progress)
        self._unload(slot)
        self.log("Loading %s %s on %s" % (slot.kind, display_name(card), hardware.describe_device(device, self.hw)))
        family = importlib.import_module("zoo.families." + card.family)
        build = family.Classifier if slot.kind == "classifier" else family.Model
        try:
            slot.model = build(card, path, device)
        except Exception as e:
            if device == "cpu":
                raise
            self.log("WARNING: could not load %s on %s (%s) -> trying CPU." % (card.name, device, _short(e)))
            device = "cpu"
            slot.model = build(card, path, device)
        slot.card, slot.device = card, device
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
            self.log("WARNING: %s failed on %s (%s) -> retrying on CPU." % (card.name, slot.device, _short(e)))
            self._unload(slot)
            self._load(slot, card, device="cpu")
            return run(slot.model)

    def predict(self, image_rgb, threshold, iou, prompt=None):
        """Detect. `prompt` is the text prompt for models that take one (card.text_prompt), ignored otherwise."""
        prompt = ((prompt or "").strip() or None) if self.card.text_prompt else None
        t = time.time()
        dets = self._on_cpu_if_gpu_fails(self.detector, lambda m: m.predict(image_rgb, threshold, iou, prompt))
        h, w = image_rgb.shape[:2]
        for d in dets:                                              # keep boxes inside the image
            d.x1, d.x2 = (min(max(v, 0), w - 1) for v in (d.x1, d.x2))
            d.y1, d.y2 = (min(max(v, 0), h - 1) for v in (d.y1, d.y2))
        return dets, time.time() - t

    def classify(self, image_rgb, dets, classes=None):
        """Give every detection a taxon with the loaded classifier (if any). Returns the seconds it took."""
        if self.classifier.model is None or not dets:
            return 0.0
        t = time.time()
        self._on_cpu_if_gpu_fails(self.classifier, lambda m: m.classify(image_rgb, dets, classes))
        return time.time() - t

    def run_file(self, path, threshold, iou, out_dir, prompt=None, classes=None):
        image = results.load_image(path)
        dets, secs = self.predict(image, threshold, iou, prompt)
        secs += self.classify(image, dets, classes)
        files = results.save(path, image, dets, out_dir, self.card.name,
                             self.classifier.card.name if self.classifier.card else "")
        return image, dets, secs, files


def _is_gpu_error(e):
    text = "%s %s" % (type(e).__name__, e)
    return any(k in text for k in ("OutOfMemory", "out of memory", "CUDA", "cuda", "MPS", "mps", "cuDNN"))


def _short(e):
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
