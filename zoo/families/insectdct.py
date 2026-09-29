"""insectDCT detectors (YOLO11, Ultralytics) - https://github.com/kimbjerge/insectDCT"""

from ..results import Detection


class Model:
    def __init__(self, card, weights_path, device):
        from ultralytics import YOLO
        self.card = card
        self.device = device
        self.yolo = YOLO(weights_path)      # image size comes from the training settings stored in the weights

    def predict(self, image_rgb, threshold, iou, prompt=None):     # no text prompt for this family
        r = self.yolo.predict(image_rgb[:, :, ::-1], conf=threshold, iou=iou, device=self.device, verbose=False)[0]
        dets = []
        for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = r.names.get(int(cls), self.card.label) if len(r.names) > 1 else self.card.label
            dets.append(Detection(x1, y1, x2, y2, conf, label))
        return dets
