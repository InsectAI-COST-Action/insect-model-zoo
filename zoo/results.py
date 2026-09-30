"""Common output for every model: a list of Detection -> annotated image, CSV and COCO JSON (with the outlines of
segmentation models)."""

import csv
import os
from dataclasses import dataclass

import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif")
CSV_FIELDS = ["image", "model", "x1", "y1", "x2", "y2", "confidence", "label",
              "classifier", "taxon", "taxon_score", "taxon_rank", "angle"]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float         # the detector's score; None when there was no detector (whole-image mode)
    label: str
    polygon: list = None      # [[x, y], ...] outline in image pixels, for segmentation models
    taxon: str = ""           # set by a classifier: what the insect is ("Unsure" if the classifier is not sure)
    taxon_score: float = None  # classifier confidence 0-1
    taxon_rank: str = ""      # e.g. species / family / order, when the classifier knows it
    taxon_options: list = None  # [(name, score, rank), ...] deepest first: broader names to fall back on (BioCLIP)
    angle: float = None       # oriented boxes: long side vs the image x-axis, degrees counter-clockwise (-90, 90];
                              # the 4 corners are in polygon, x1..y2 is the upright hull. None = an upright box

    def row(self, image, model, classifier=""):
        return {"image": image, "model": model, "x1": round(self.x1), "y1": round(self.y1), "x2": round(self.x2),
                "y2": round(self.y2),
                "confidence": "" if self.confidence is None else round(self.confidence, 4), "label": self.label,
                "classifier": classifier, "taxon": self.taxon,
                "taxon_score": "" if self.taxon_score is None else round(self.taxon_score, 4),
                "taxon_rank": self.taxon_rank, "angle": "" if self.angle is None else round(self.angle, 2)}

    def caption(self):
        if self.taxon and self.taxon != "Unsure":
            return "%s %.2f" % (self.taxon, self.taxon_score or 0)
        unsure = " (unsure)" if self.taxon == "Unsure" else ""
        if not self.label:                       # e.g. a whole-image box (classifier only): no detector label
            return "Unsure" if unsure or self.confidence is None else "%.2f" % self.confidence
        return "%s %.2f%s" % (self.label, self.confidence, unsure)


def load_image(path):
    """RGB uint8 array, EXIF rotation applied (phones/cameras often store images sideways)."""
    from PIL import Image, ImageOps
    if path.lower().endswith((".heic", ".heif")):
        from pi_heif import register_heif_opener
        register_heif_opener()
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))          # a writable copy


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder, f)))


def draw(image_rgb, detections):
    """Boxes (and outlines, if any) with confidence, drawn on a copy of the image. Returns RGB."""
    import cv2
    img = np.ascontiguousarray(image_rgb[:, :, ::-1])            # OpenCV draws in BGR
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 800))
    font = max(0.5, max(h, w) / 2000)
    box_color, outline_color = (40, 200, 40), (255, 190, 0)     # green boxes, cyan-ish outlines (BGR)
    for d in detections:
        p1, p2 = (int(round(d.x1)), int(round(d.y1))), (int(round(d.x2)), int(round(d.y2)))
        if d.angle is not None and d.polygon is not None:          # oriented box: draw the rotated box itself
            corners = np.round(np.asarray(d.polygon)).astype(np.int32)
            cv2.polylines(img, [corners], True, box_color, thick)
            top = corners[corners[:, 1].argmin()]
            p1, p2 = (int(top[0]), int(top[1])), (int(top[0]), int(corners[:, 1].max()))
        else:
            if d.polygon is not None and len(d.polygon) > 2:
                cv2.polylines(img, [np.round(np.asarray(d.polygon)).astype(np.int32)], True, outline_color, thick)
            cv2.rectangle(img, p1, p2, box_color, thick)
        text = d.caption()
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p2[1] + th + 4
        cv2.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw + 4, ty + base), box_color, -1)
        cv2.putText(img, text, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick, cv2.LINE_AA)
    return img[:, :, ::-1]


def save(image_path, image_rgb, detections, out_dir, model_name, classifier_name="", meta=None):
    """Write <stem>_annotated.jpg, <stem>_detections.csv, <stem>_coco.json (COCO, e.g. to train a model on) and
    <stem>_isir.json (ISIR, the InsectAI intermediate representation). meta: {"config": settings, "device": ...}.
    Returns the list of written file paths."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    name = os.path.basename(image_path)
    files = []

    out_img = os.path.join(out_dir, stem + "_annotated.jpg")
    ok, buf = cv2.imencode(".jpg", draw(image_rgb, detections)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(out_img)                                    # tofile works with non-ASCII paths on Windows
        files.append(out_img)

    out_csv = os.path.join(out_dir, stem + "_detections.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(d.row(name, model_name, classifier_name) for d in detections)
    files.append(out_csv)

    from .export import Entry, isir, write_coco, write_isir
    h, w = image_rgb.shape[:2]
    entry = Entry(image_path, name, w, h, detections)
    files.append(write_coco(os.path.join(out_dir, stem + "_coco.json"), [entry], model_name, classifier_name))
    meta = meta or {}
    files.append(write_isir(os.path.join(out_dir, stem + "_isir.json"),
                            isir(entry, model_name, classifier_name, meta.get("config"), meta.get("device", ""))))
    return files


def write_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
