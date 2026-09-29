"""Load a model from the zoo (download weights -> hardware check -> load) and run it; shared by the CLI and the UI."""

import gc
import importlib
import time

from . import hardware, results
from .weights import ensure_weights


class Zoo:
    """Keeps one model loaded at a time, so switching models in the UI does not fill up GPU memory."""

    def __init__(self, device="auto", log=print):
        self.hw = hardware.probe()
        self.requested_device = device
        self.log = log
        self.card = None
        self.model = None
        self.device = None

    def load(self, card, progress=None):
        """progress(done, total, message) is forwarded to the weight download."""
        seen = len(self.hw.notes)
        device = hardware.pick_device(self.requested_device, self.hw)
        device, warnings = hardware.check_model(card, device, self.hw)
        for w in self.hw.notes[seen:] + warnings:
            self.log("WARNING: " + w)
        if self.card is card and self.device == device:
            return self.model

        path = ensure_weights(card, progress)
        self.unload()
        self.log("Loading %s on %s" % (card.name, hardware.describe_device(device, self.hw)))
        family = importlib.import_module("zoo.families." + card.family)
        try:
            self.model = family.Model(card, path, device)
        except Exception as e:
            if device == "cpu":
                raise
            self.log("WARNING: could not load %s on %s (%s) -> trying CPU." % (card.name, device, _short(e)))
            device = "cpu"
            self.model = family.Model(card, path, device)
        self.card, self.device = card, device
        return self.model

    def unload(self):
        self.model = self.card = self.device = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict(self, image_rgb, threshold, iou, prompt=None):
        """Run the loaded model; if the GPU runs out of memory (or fails), retry once on CPU.
        `prompt` is the text prompt for models that take one (card.text_prompt), ignored otherwise."""
        prompt = ((prompt or "").strip() or None) if self.card.text_prompt else None
        t = time.time()
        try:
            dets = self.model.predict(image_rgb, threshold, iou, prompt)
        except Exception as e:
            if self.device == "cpu" or not _is_gpu_error(e):
                raise
            card = self.card
            self.log("WARNING: %s failed on %s (%s) -> retrying on CPU." % (card.name, self.device, _short(e)))
            self.requested_device = "cpu"
            self.unload()
            self.load(card)
            dets = self.model.predict(image_rgb, threshold, iou, prompt)
        return dets, time.time() - t

    def run_file(self, path, threshold, iou, out_dir, prompt=None):
        image = results.load_image(path)
        dets, secs = self.predict(image, threshold, iou, prompt)
        files = results.save(path, image, dets, out_dir, self.card.name)
        return image, dets, secs, files


def _is_gpu_error(e):
    text = "%s %s" % (type(e).__name__, e)
    return any(k in text for k in ("OutOfMemory", "out of memory", "CUDA", "cuda", "MPS", "mps", "cuDNN"))


def _short(e):
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
