"""Exports of the results: COCO JSON (to train models on) and a Camtrap DP data package (to share / publish camera-trap
data, e.g. on GBIF).

Both take a list of Entry: one image with its size and detections (see results.Detection)."""

import csv
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .registry import REPO_URL

CAMTRAP_DP = "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/"
DETECTOR_LABEL_TAXA = {"insect": ("Insecta", "class"), "arthropod": ("Arthropoda", "phylum")}   # no classifier
CAPTURE_METHODS = ("activityDetection", "timeLapse")
SAMPLING_DESIGNS = ("simpleRandom", "systematicRandom", "clusteredRandom", "experimental", "targeted", "opportunistic")


@dataclass
class Entry:
    path: str          # the image file
    file_name: str     # the name written to the export (relative to the input folder)
    width: int
    height: int
    detections: list


def category(d):
    """COCO category of a detection: what the classifier says, else what the detector found."""
    return d.taxon if d.taxon and d.taxon != "Unsure" else (d.label or "object")


# --------------------------------------------------------------------------------------------------- COCO
def coco(entries, detector, classifier=""):
    """COCO dataset (images, annotations, categories), so the results can be used as (pre-)labels for training.
    bbox = [x, y, width, height] in pixels; outlines of segmentation models as `segmentation` polygons. Extra fields
    per annotation: score (detector), label, taxon, taxon_score, taxon_rank."""
    names = sorted({category(d) for e in entries for d in e.detections})
    ids = {n: i for i, n in enumerate(names, 1)}
    images, annotations = [], []
    for image_id, e in enumerate(entries, 1):
        images.append({"id": image_id, "file_name": e.file_name, "width": e.width, "height": e.height})
        for d in e.detections:
            x, y, w, h = d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1
            ann = {"id": len(annotations) + 1, "image_id": image_id, "category_id": ids[category(d)],
                   "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2),
                   "iscrowd": 0, "score": round(d.confidence, 4), "label": d.label, "taxon": d.taxon,
                   "taxon_score": None if d.taxon_score is None else round(d.taxon_score, 4),
                   "taxon_rank": d.taxon_rank}
            if d.polygon is not None and len(d.polygon) > 2:
                ann["segmentation"] = [[round(float(v), 1) for point in d.polygon for v in point]]
                ann["area"] = round(_polygon_area(d.polygon), 2)
            annotations.append(ann)
    return {"info": {"description": "Predictions of the InsectAI model zoo", "url": REPO_URL,
                     "date_created": _now(), "detector": detector, "classifier": classifier},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": ids[n], "name": n, "supercategory": ""} for n in names]}


def write_coco(path, entries, detector, classifier=""):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco(entries, detector, classifier), f, indent=1)
    return path


def _polygon_area(poly):
    xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------------------------------- Camtrap DP
def write_camtrapdp(folder, entries, detector, classifier="", info=None, name_map=None, log=print):
    """Camtrap DP 1.0.2 data package (https://camtrap-dp.tdwg.org): datapackage.json + deployments.csv, media.csv,
    observations.csv in `folder`. One deployment (camera / site) for all images; one observation per detection
    (observationLevel media); an image without detections gets one 'blank' observation.

    info: project, contributor, deployment_id, latitude, longitude, capture_method, sampling_design (main.py).
    name_map(taxon) -> (scientific name, rank, comment) or None (not an animal): turns a classifier's own class names
    into scientific names (e.g. insectDCT's 'Aranaea' -> 'Araneae'). Returns the list of problems (empty = valid)."""
    info = dict(info or {})
    os.makedirs(folder, exist_ok=True)
    problems = []
    created = _now()
    by = "InsectAI model zoo: %s%s" % (detector, " + " + classifier if classifier else "")

    deployment = info.get("deployment_id") or _deployment_name(entries)
    stamps = [_timestamp(e.path) for e in entries]
    gps = next((g for g in (_gps(e.path) for e in entries) if g), None)
    lat, lon = info.get("latitude"), info.get("longitude")
    if (lat is None or lon is None) and gps:
        lat, lon = gps
        log("Camtrap DP: camera position from the photos' GPS: %.5f, %.5f" % (lat, lon))
    for key, allowed in (("capture_method", CAPTURE_METHODS), ("sampling_design", SAMPLING_DESIGNS)):
        if info.get(key) not in allowed:
            problems.append("%s = %r is not allowed; use one of: %s" % (key, info.get(key), ", ".join(allowed)))
    if lat is None or lon is None:
        problems.append("latitude/longitude of the camera are missing (required): give --latitude and --longitude "
                        "(or set them in CAMTRAPDP_INFO in main.py) and run again, or fill them in deployments.csv")
    if any(s[1] for s in stamps):
        log("Camtrap DP: %d photo(s) have no date in their EXIF data; used the file's modification time instead."
            % sum(1 for s in stamps if s[1]))

    media, observations, taxa, left_out = [], [], {}, 0
    for n, (e, (stamp, _)) in enumerate(zip(entries, stamps), 1):
        media_id = "m%05d" % n
        media.append({"mediaID": media_id, "deploymentID": deployment, "captureMethod": info.get("capture_method"),
                      "timestamp": stamp, "filePath": Path(os.path.abspath(e.path)).as_uri(), "filePublic": False,
                      "fileName": os.path.basename(e.path), "fileMediatype": _mediatype(e.path)})
        kept = 0
        for d in e.detections:
            name, rank, comment, probability = _scientific(d, name_map)
            if name is False:                                        # the classifier says: not an animal
                left_out += 1
                continue
            kept += 1
            if name:
                taxa.setdefault(name, rank)
            observations.append({
                "observationID": "o%06d" % (len(observations) + 1), "deploymentID": deployment,
                "mediaID": media_id, "eventStart": stamp, "eventEnd": stamp, "observationLevel": "media",
                "observationType": "animal" if name else "unclassified", "scientificName": name, "count": 1,
                "bboxX": _clip(d.x1 / e.width), "bboxY": _clip(d.y1 / e.height),
                "bboxWidth": _clip((d.x2 - d.x1) / e.width, 1e-6), "bboxHeight": _clip((d.y2 - d.y1) / e.height, 1e-6),
                "classificationMethod": "machine", "classifiedBy": by, "classificationTimestamp": created,
                "classificationProbability": round(probability, 4) if probability is not None else None,
                "observationComments": comment})
        if not kept:
            observations.append({"observationID": "o%06d" % (len(observations) + 1), "deploymentID": deployment,
                                 "mediaID": media_id, "eventStart": stamp, "eventEnd": stamp,
                                 "observationLevel": "media", "observationType": "blank",
                                 "classificationMethod": "machine", "classifiedBy": by,
                                 "classificationTimestamp": created})
    if left_out:
        log("Camtrap DP: %d detection(s) the classifier called vegetation (not an animal) are left out." % left_out)

    times = sorted(s for s, _ in stamps)
    deployments = [{"deploymentID": deployment, "locationName": deployment, "latitude": lat, "longitude": lon,
                    "deploymentStart": times[0], "deploymentEnd": times[-1],
                    "deploymentComments": "Start / end are the first / last photo (the camera may have run longer)."}]
    _write_table(os.path.join(folder, "deployments.csv"), "deployments", deployments)
    _write_table(os.path.join(folder, "media.csv"), "media", media)
    _write_table(os.path.join(folder, "observations.csv"), "observations", observations)

    contributors = [{"title": "InsectAI model zoo", "path": REPO_URL, "role": "contributor"}]
    if info.get("contributor"):
        contributors.insert(0, {"title": info["contributor"], "role": "contact"})
    package = {
        "profile": CAMTRAP_DP + "camtrap-dp-profile.json",
        "name": _slug("insect-model-zoo-" + deployment), "id": str(uuid.uuid4()), "created": created,
        "title": info.get("project") or "Insect camera trap",
        "contributors": contributors,
        "sources": [{"title": "InsectAI model zoo", "path": REPO_URL}],
        "project": {"title": info.get("project") or "Insect camera trap",
                    "description": "Detections by %s%s, made with the InsectAI model zoo." % (
                        detector, ", classified by " + classifier if classifier else ""),
                    "samplingDesign": info.get("sampling_design") or "targeted",
                    "captureMethod": [info.get("capture_method") or "timeLapse"],
                    "individualAnimals": False, "observationLevel": ["media"]},
        "temporal": {"start": times[0][:10], "end": times[-1][:10]},
        "taxonomic": [dict(scientificName=n, **({"taxonRank": r} if r else {})) for n, r in sorted(taxa.items())],
        "resources": [{"name": t, "path": t + ".csv", "profile": "tabular-data-resource", "format": "csv",
                       "mediatype": "text/csv", "encoding": "utf-8", "schema": CAMTRAP_DP + t + "-table-schema.json"}
                      for t in ("deployments", "media", "observations")],
    }
    if lat is not None and lon is not None:
        package["spatial"] = {"type": "Point", "coordinates": [lon, lat]}
    with open(os.path.join(folder, "datapackage.json"), "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)
    return problems


FIELDS = {
    "deployments": ["deploymentID", "locationID", "locationName", "latitude", "longitude", "coordinateUncertainty",
                    "deploymentStart", "deploymentEnd", "setupBy", "cameraID", "cameraModel", "cameraDelay",
                    "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading", "detectionDistance",
                    "timestampIssues", "baitUse", "featureType", "habitat", "deploymentGroups", "deploymentTags",
                    "deploymentComments"],
    "media": ["mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic", "fileName",
              "fileMediatype", "exifData", "favorite", "mediaComments"],
    "observations": ["observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd",
                     "observationLevel", "observationType", "cameraSetupType", "scientificName", "count", "lifeStage",
                     "sex", "behavior", "individualID", "individualPositionRadius", "individualPositionAngle",
                     "individualSpeed", "bboxX", "bboxY", "bboxWidth", "bboxHeight", "classificationMethod",
                     "classifiedBy", "classificationTimestamp", "classificationProbability", "observationTags",
                     "observationComments"],
}


def _write_table(path, table, rows):
    """All the table's columns in the order of the Camtrap DP schema; empty = not known."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS[table])
        wr.writeheader()
        for row in rows:
            wr.writerow({k: _cell(row.get(k)) for k in FIELDS[table]})


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def _scientific(d, name_map):
    """(scientific name, rank, comment, probability); name False = not an animal (left out); "" = unknown."""
    if d.taxon and d.taxon != "Unsure":
        mapped = name_map(d.taxon) if name_map else (d.taxon, d.taxon_rank, "")
        if mapped is None:
            return False, "", "", None
        name, rank, comment = mapped
        return name, rank or d.taxon_rank, comment, d.taxon_score
    name, rank = DETECTOR_LABEL_TAXA.get((d.label or "").lower(), ("", ""))
    comment = "classifier unsure" if d.taxon == "Unsure" else ""
    if not name and d.label:
        comment = (comment + "; " if comment else "") + "detector label: " + d.label
    return name, rank, comment, d.confidence


def _clip(v, low=0.0):
    return round(min(1.0, max(low, v)), 6)


def _timestamp(path):
    """(ISO 8601 time with UTC offset, fallback used): EXIF DateTimeOriginal (+ its offset if stored, else this
    computer's time zone), else the file's modification time."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif().get_ifd(0x8769)                      # Exif IFD
        raw, offset = exif.get(36867), exif.get(36881)              # DateTimeOriginal, OffsetTimeOriginal
        if raw:
            t = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
            if offset:
                return datetime.fromisoformat(t.isoformat() + str(offset).strip()).isoformat(), False
            return t.astimezone().isoformat(), False                # no offset stored: this computer's time zone
    except Exception:
        pass
    return datetime.fromtimestamp(os.path.getmtime(path)).astimezone().replace(microsecond=0).isoformat(), True


def _gps(path):
    """(latitude, longitude) from the photo's EXIF GPS, or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.getexif().get_ifd(0x8825)                         # GPS IFD
        if 2 in g and 4 in g:
            def deg(v, ref):
                d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
                return -d if ref in ("S", "W") else d
            return round(deg(g[2], g.get(1, "N")), 6), round(deg(g[4], g.get(3, "E")), 6)
    except Exception:
        pass
    return None


def _mediatype(path):
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".tif": "image/tiff",
            ".tiff": "image/tiff", ".bmp": "image/bmp", ".webp": "image/webp", ".heic": "image/heic",
            ".heif": "image/heif"}.get(ext, "image/" + ext.lstrip("."))


def _deployment_name(entries):
    folders = {os.path.basename(os.path.dirname(os.path.abspath(e.path))) for e in entries}
    return folders.pop() if len(folders) == 1 else "deployment"


def _slug(text):
    import re
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-") or "insect-model-zoo"
