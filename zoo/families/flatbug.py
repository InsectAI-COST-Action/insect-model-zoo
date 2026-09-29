"""flat-bug arthropod detection + segmentation (YOLOv8/YOLO26-seg with pyramid tiling) -
https://github.com/darsa-group/flat-bug"""

from ..results import Detection


class Model:
    def __init__(self, card, weights_path, device):
        import torch
        from flat_bug.predictor import Predictor
        self.card = card
        self.device = device
        self.predictor = Predictor(model=weights_path, device=device, dtype=torch.float32)

    def predict(self, image_rgb, threshold, iou, prompt=None):     # no text prompt for this family
        import torch
        self.predictor.set_hyperparameters(SCORE_THRESHOLD=threshold, OVERLAP_THRESHOLD=iou)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous()          # HWC -> CHW, uint8
        # flat-bug wraps its NMS helpers in torch.compile. That needs Triton (not available on Windows/macOS) or a
        # C++ compiler, and the first call can take minutes, so run them as plain PyTorch instead - same results.
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            out = self.predictor.pyramid_predictions(tensor, path="image.jpg")
        if len(out) == 0:
            return []
        boxes = out.boxes.float().cpu().tolist()
        confs = out.confs.float().cpu().tolist()
        contours = out.contours
        dets = []
        for i, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, confs)):
            poly = contours[i].float().cpu().tolist() if i < len(contours) else None           # [[x, y], ...]
            dets.append(Detection(x1, y1, x2, y2, conf, self.card.label, poly))
        return dets
