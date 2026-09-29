"""SAM 3 (Meta): finds and outlines whatever the text prompt describes, run through Ultralytics -
https://github.com/facebookresearch/sam3"""

from ..results import Detection


class Model:
    def __init__(self, card, weights_path, device):
        from ultralytics.models.sam import SAM3SemanticPredictor
        self.card = card
        self.device = device
        overrides = dict(task="segment", mode="predict", model=weights_path, device=device, imgsz=1008,  # native size
                         conf=card.default_threshold, iou=card.default_iou, verbose=False, save=False)
        if device.startswith("cuda"):
            overrides["quantize"] = 16                   # half precision: half the GPU memory, same results
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.predictor.setup_model(verbose=False)        # load now, so memory problems show up before the first image

    def predict(self, image_rgb, threshold, iou, prompt=None):
        prompts = [p.strip() for p in (prompt or self.card.default_prompt).split(",") if p.strip()]
        self.predictor.args.conf, self.predictor.args.iou = threshold, iou
        self.predictor.set_image(image_rgb[:, :, ::-1].copy())                   # Ultralytics wants BGR
        r = self.predictor(text=prompts)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        polygons = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        dets = []
        for (x1, y1, x2, y2), conf, cls, poly in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                                     r.boxes.cls.tolist(), polygons):
            label = prompts[int(cls)] if int(cls) < len(prompts) else prompts[0]
            dets.append(Detection(x1, y1, x2, y2, conf, label, _simplify(poly)))
        return dets


def _simplify(poly, tolerance=1.0):
    """Mask outlines come pixel by pixel; keep the shape within 1 px with far fewer points."""
    import cv2
    import numpy as np
    if poly is None or len(poly) < 3:
        return None
    return cv2.approxPolyDP(np.asarray(poly, np.float32).reshape(-1, 1, 2), tolerance, True).reshape(-1, 2).tolist()
