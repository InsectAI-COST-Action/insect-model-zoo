"""BioCLIP (Imageomics): zero-shot classification with names you choose, via open_clip -
https://github.com/Imageomics/bioclip-2

Each insect is cropped from the image and compared with a text for every class name; the best match wins.
Default names are arthropod orders; give your own (species, genera, common names) for finer classes.
"""

import numpy as np

ARCHITECTURE = {"bioclip-2.5": "ViT-H-14", "bioclip-2": "ViT-L-14"}     # open_clip names of the two backbones
MARGIN = 0.1            # crop 10% around each box: a bit of context helps CLIP models


class Classifier:
    def __init__(self, card, weights_path, device):
        import open_clip
        self.card = card
        self.device = device
        arch = ARCHITECTURE[card.name]
        precision = "fp16" if device.startswith("cuda") else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=weights_path, device=device, precision=precision)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self._text = (None, None)                                   # (class list, text features) cache

    def _text_features(self, pairs):
        import torch
        if self._text[0] != pairs:
            with torch.inference_mode():
                feats = self.model.encode_text(self.tokenizer([text for _, text in pairs]).to(self.device))
                self._text = (pairs, feats / feats.norm(dim=-1, keepdim=True))
        return self._text[1]

    def classify(self, image_rgb, detections, classes=None):
        """classes: your own names (list of str); None/empty = the default names of the model card."""
        import torch
        from PIL import Image
        if not detections:
            return
        own = [c.strip() for c in (classes or []) if c and c.strip()]
        pairs = tuple((c, "a photo of %s." % c) for c in own) if own else self.card.classes
        text = self._text_features(pairs)

        h, w = image_rgb.shape[:2]
        crops = []
        for d in detections:
            mx, my = (d.x2 - d.x1) * MARGIN, (d.y2 - d.y1) * MARGIN
            x1, y1 = max(0, int(d.x1 - mx)), max(0, int(d.y1 - my))
            x2, y2 = min(w, int(d.x2 + mx) + 1), min(h, int(d.y2 + my) + 1)
            crop = image_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((8, 8, 3), np.uint8)
            crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(crop))))

        dtype = self.model.visual.conv1.weight.dtype          # fp16 on GPU, fp32 on CPU
        for start in range(0, len(crops), 32):
            batch = torch.stack(crops[start:start + 32]).to(self.device, dtype)
            with torch.inference_mode():
                img = self.model.encode_image(batch)
                img = img / img.norm(dim=-1, keepdim=True)
                probs = (self.model.logit_scale.exp() * img @ text.T).float().softmax(dim=-1).cpu().numpy()
            for d, p in zip(detections[start:start + 32], probs):
                best = int(p.argmax())
                d.taxon, d.taxon_score = pairs[best][0], float(p[best])
                d.taxon_rank = "" if own else "order"
