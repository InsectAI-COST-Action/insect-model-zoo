"""Grounding DINO (IDEA Research): finds what you describe in words, with boxes - via Hugging Face transformers,
https://github.com/IDEA-Research/GroundingDINO

Like SAM 3 it takes a text prompt ("bee, butterfly"), but it gives boxes only (no outlines), is much smaller and is not
gated. Each box gets exactly one of your words as its label: the score of a word is the mean of its tokens' scores
(as in mmdetection), not transformers' text labels, which can glue several words together ('bee butterfly'). DETR-style
models do no NMS of their own, so overlapping boxes are merged here (class-agnostic: one box per insect)."""

import os

from ..results import Detection


def _fix_text_position_ids():
    """transformers >= 5.9 passes the text position ids as int64 into a sin/cos embedding (huggingface/transformers
    issue 47674): cast them to float first. Does nothing once transformers fixes it."""
    from transformers.models.grounding_dino import modeling_grounding_dino as gd
    layer = gd.GroundingDinoEncoderLayer
    if getattr(layer, "_zoo_fixed", False):
        return
    original = layer.get_text_position_embeddings

    def fixed(self, text_features, text_position_embedding, text_position_ids):
        if text_position_ids is not None and not text_position_ids.is_floating_point():
            text_position_ids = text_position_ids.to(text_features.dtype)
        return original(self, text_features, text_position_embedding, text_position_ids)

    layer.get_text_position_embeddings, layer._zoo_fixed = fixed, True


class Model:
    def __init__(self, card, weights_path, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.utils import logging
        logging.disable_progress_bar()
        _fix_text_position_ids()
        folder = os.path.dirname(weights_path)          # model.safetensors + its config / tokenizer files
        self.card, self.device = card, device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            folder, local_files_only=True, dtype=self.dtype).to(device).eval()
        self.dot = self.processor.tokenizer.convert_tokens_to_ids(".")

    def predict(self, image_rgb, threshold, iou, prompt=None):
        import torch
        from torchvision.ops import nms
        words = [p.strip().lower().replace(".", " ").strip()
                 for p in (prompt or self.card.default_prompt).replace(";", ",").split(",")]
        words = list(dict.fromkeys(p for p in words if p))             # unique, in the order given
        text = ". ".join(words) + "."                                   # the model's format: "bee. butterfly."
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, text=text, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            out = self.model(**inputs)
        ids = inputs["input_ids"][0].tolist()
        spans, current = [], []                         # the token positions of each word, between the dots
        for i, token in enumerate(ids[1:-1], start=1):
            if token == self.dot:
                spans.append(current)
                current = []
            else:
                current.append(i)
        spans = spans[:len(words)]                      # a very long prompt is cut at 256 tokens
        if not spans:
            return []
        probs = out.logits[0].float().sigmoid()         # (900 queries, 256 tokens)
        per_word = torch.stack([probs[:, s].mean(-1) for s in spans], -1)
        scores, which = per_word.max(-1)
        keep = scores > threshold
        cx, cy, bw, bh = out.pred_boxes[0].float()[keep].unbind(-1)    # normalised centre x, y, width, height
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        scores, which = scores[keep], which[keep]
        k = nms(boxes, scores, iou)
        return [Detection(*box, score, words[c]) for box, score, c in
                zip(boxes[k].tolist(), scores[k].tolist(), which[k].tolist())]
