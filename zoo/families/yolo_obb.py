"""Ultralytics YOLO detectors with ORIENTED (rotated) boxes, e.g. Mothbot Detect -
https://github.com/Digital-Naturalism-Laboratories/Mothbot_Process

Each insect gets a box turned to fit its body. The zoo keeps the box as its 4 corners (Detection.polygon), its
upright hull as x1, y1, x2, y2, and its angle (Detection.angle, see results.py)."""

import math

from ..results import Detection

MAX_DET = 10000          # a light-trap photo can hold thousands of insects (Ultralytics' default is 300)


class Model:
    def __init__(self, card, weights_path, device):
        from ultralytics import YOLO
        self.card = card
        self.device = device
        self.yolo = YOLO(weights_path)      # task 'obb' and image size (1600 for Mothbot) come from the weights
        _lift_nms_time_limit()

    def predict(self, image_rgb, threshold, iou, prompt=None):     # no text prompt for this family
        r = self.yolo.predict(image_rgb[:, :, ::-1], conf=threshold, iou=iou, max_det=MAX_DET, device=self.device,
                              verbose=False)[0]
        dets = []
        if r.obb is None:
            return dets
        for hull, corners, conf, cls in zip(r.obb.xyxy.tolist(), r.obb.xyxyxyxy.tolist(), r.obb.conf.tolist(),
                                            r.obb.cls.tolist()):
            label = self.card.label           # one label for every find ("insect"); naming is the classifier's job
            dets.append(Detection(*hull, conf, label, polygon=[[float(x), float(y)] for x, y in corners],
                                  angle=long_side_angle(corners)))
        return dets


def long_side_angle(corners):
    """Angle of the box's long side from the image x-axis, in degrees counter-clockwise (as seen on screen),
    -90 < angle <= 90; 0 = lying flat. Taken from the corners, so it does not depend on how the model stores
    its own angle (Ultralytics' raw OBB angle is in radians and its width/height order is not fixed)."""
    (x0, y0), (x1, y1), (x2, y2) = corners[:3]
    a, b = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1)
    dx, dy = a if math.hypot(*a) >= math.hypot(*b) else b
    angle = math.degrees(math.atan2(-dy, dx))           # image y points down: minus = counter-clockwise
    if angle <= -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    return round(angle, 2)


def _lift_nms_time_limit():
    """Ultralytics gives NMS a time budget per batch (about 2 s) and, when rotated NMS on thousands of boxes takes
    longer, returns the image with no detections at all (only a log warning). Default the budget to effectively
    unlimited, as upstream Mothbot does. Idempotent."""
    from ultralytics.utils import nms
    if getattr(nms.non_max_suppression, "_zoo_patched", False):
        return
    original = nms.non_max_suppression

    def patched(*args, **kwargs):
        kwargs.setdefault("max_time_img", 3600.0)
        return original(*args, **kwargs)

    patched._zoo_patched = True
    nms.non_max_suppression = patched
