# Golden JSON format (draft)

> [!NOTE]
> **Draft, not produced by the zoo yet.** Written up from the team's whiteboard sketch, with proposed additions and
> open questions. Once the open decisions below are settled, the zoo can write it (`<image>_golden.json`, plus
> `golden.jsonl` for a folder) with a JSON Schema to check it.

One JSON object per image: the image, the models that ran, where and when the photo was taken, and a list of single
instances (detections or human annotations).

**Type notation from the board:** `s` string, `i` integer, `f` float, `s|i` string or integer, `[4f]` four floats,
`[s]` list of strings, `obj` object, `iso-DT` ISO 8601 date-time, `uuidv4`; `?` = optional.

## The board, and where the zoo fills each key

| Key | Type | Where the zoo gets it |
|---|---|---|
| **Single instance** (one per detection, in `instances`) | | |
| `id` | s\|i | running number within the image (0, 1, 2, ...) |
| `bbox` | [4f] | the detector's box, in pixels on the upright image (after EXIF rotation) |
| `conf` | ? f | the detector's confidence |
| `angle` | ? f | none of our models predicts rotated boxes: 0 / left out for now |
| **Global image information** | | |
| `image_width`, `image_height` | i | size of the decoded image, after EXIF rotation, so boxes and size match |
| `image_id` | s\|i | path relative to the input folder, e.g. `site_A/IMG_0042.JPG` |
| `image_format` | ? s | the file's format as Pillow reads it: `JPEG`, `PNG`, `HEIF` |
| **Model / inference metadata** | | |
| `model_name` | ? s | model name from `zoo/registry.py`, e.g. `insectdct-v8-m` |
| `model_uuid` | ? uuidv4 | *new:* a fixed uuid4 per model, generated once and stored in `zoo/registry.py` |
| `prediction_timestamp` | ? iso-DT | time of the run (UTC) |
| `prediction_config` | ? obj | the settings used: thresholds, IoU, prompt, names, device, zoo commit |
| **Context metadata** | | |
| `image_timestamp` | ? iso-DT | EXIF `DateTimeOriginal` with the photo's own `OffsetTimeOriginal`; if the photo stores no zone, the camera's zone from the settings (else this computer's), marked `exif_zone_assumed`; left out if the photo has no date (the file date goes to `extra_information`) |
| `latitude`, `longitude` | ? f | `--latitude` / `--longitude` if given, else the photo's EXIF GPS (already read for Camtrap DP) |
| `crs` | ? s\|i | `EPSG:4326` whenever a position is given (EXIF GPS is WGS 84) |
| `tag_list` | ? [s] | *new* `--tags` option, optionally the Windows "Tags" EXIF field (XPKeywords); no XMP or OCR |
| `group_name` | ? s | camera / site: folder name or `--deployment_id` (same as the Camtrap DP deployment) |
| `group_id` | ? i | a number you give (*new* `--group_id`) |
| **Versioning** | | |
| `ir_version`, `ir_version_id` | s, i | constants in the code, e.g. `"1.0.0"` and `1` |
| **Extra** | | |
| `extra_information` | ? obj | anything else: EXIF camera make / model / serial, file date as fallback, ... |

## Proposed additions

Not on the board. Needed for the zoo's own output, or to read the rest unambiguously:

- **Per instance:** `label` (what the detector calls it), `taxon`, `taxon_conf`, `taxon_rank` (the classifier's answer:
  the board has no class at all), `taxon_options` (BioCLIP's best name per rank), `polygon` (outlines of segmentation
  models), `source` (`prediction` or `annotation`, so human labels for training fit the same format) and a per-instance
  `extra_information`.
- **Per image:** `bbox_format` and `angle_unit` (what the four numbers and the angle mean), `image_sha256` (same photo =
  same hash), `image_exif_orientation`, `models` (detector *and* classifier, with weight checksums and licenses),
  `prediction_runtime_s`, `image_timestamp_source` (`exif` or `exif_zone_assumed`) and `location_source` (how far
  to trust them),
  `coordinate_uncertainty_m`.

## Open decisions

1. **IDs:** `id`, `image_id` and `group_id` mix `s`, `i` and `s|i`. Proposal: always strings.
2. **`bbox` / `angle`:** which four numbers, and the angle's unit and direction. Proposal: centre x, centre y, width,
   height in pixels (`cxcywh_px`), angle in degrees counter-clockwise, 0 = upright. (COCO style `[x_min, y_min, w, h]`
   also works, but it has to be decided: it changes every box.)
3. **Two models:** the board has one `model_name` / `model_uuid`; the zoo runs a detector and a classifier. Proposal: keep
   them for the detector and add a `models` list.
4. **`crs`:** keys called latitude / longitude only fit a geographic CRS. Proposal: always `EPSG:4326`.
5. **Small ones:** is `ir_version_id` needed next to `ir_version`; should `image_format` be a MIME type (`image/jpeg`)?

## Example

Every key filled, with made-up values. `# NEW` = proposed addition, `# ⚠` = something on the board that looks off.
The same example without comments, as valid JSON: [`golden_example.json`](golden_example.json).

```
{
  # ================= versioning (board) =================
  "ir_version": "1.0.0",
  "ir_version_id": 1,                         # ⚠ duplicates ir_version; keep only if tools need an int to compare

  # ================= global image information (board) =================
  "image_id": "meadow_cam_03/2026-07-14_10-32-05.jpg",   # ⚠ board allows s|i: mixed types break joins/sorting -> always string
  "image_width": 4056,
  "image_height": 3040,
  "image_format": "JPEG",                     # ⚠ Pillow's name; Camtrap DP / web tools expect a MIME type: "image/jpeg"

  # ----- NEW: needed to read the rest correctly -----
  "image_sha256": "4be1c07d9f2a6e83b5d0a7c1f96e24d8a3b7c5e0f1d2a9b8c7e6f5a4b3c2d1e0",   # NEW: same photo = same hash, even if renamed
  "image_exif_orientation": 1,                # NEW: 1 = stored upright (6 = sideways); size and boxes are always for the upright image
  "bbox_format": "cxcywh_px",                 # NEW: board says [4f] but not which 4 -> centre x, centre y, width, height, pixels
  "angle_unit": "deg_ccw",                    # NEW: board has angle but no unit/direction -> degrees, counter-clockwise, 0 = upright

  # ================= model / inference metadata (board) =================
  "model_name": "insectdct-v8-m",             # ⚠ board has ONE model; we run detector + classifier -> both are in "models"
  "model_uuid": "3f6c2a9e-8d41-4b7a-9c0e-5a1d27e4b8f3",  # ⚠ uuid4 is random: fine as a fixed label, but weights_sha256 is checkable
  "prediction_timestamp": "2026-09-30T12:14:14Z",
  "prediction_config": {
    "threshold": 0.407,
    "iou": 0.3,
    "prompt": "bee, hoverfly",                # only text-prompt detectors (SAM 3) use it
    "cls_threshold": 0.5,
    "classes": null,                          # no own names: BioCLIP picked from its species table (taxon_options)
    "bioclip_insect_taxa": true,
    "device": "cuda:0",
    "zoo_version": "f437fed"
  },

  # ----- NEW: every model used, not just one -----
  "models": [
    {"role": "detector", "model_name": "insectdct-v8-m", "model_uuid": "3f6c2a9e-8d41-4b7a-9c0e-5a1d27e4b8f3",
     "weights_sha256": "81f3ce9e89f2e3cf4ad6ba532f85c7c791f58e562875c8646419cfc2ee7de8e0", "license": "GPL-3.0"},
    {"role": "classifier", "model_name": "bioclip-2.5", "model_uuid": "b71e0c55-2f9a-4d3e-8a61-0c9f4e2d7a18",
     "weights_sha256": "ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97", "license": "MIT"}
  ],
  "prediction_runtime_s": 0.41,               # NEW: result, not config -> kept out of prediction_config

  # ================= context metadata (board) =================
  "image_timestamp": "2026-07-14T10:32:05+02:00",
  "image_timestamp_source": "exif",           # NEW: exif (own zone) | exif_zone_assumed (zone from settings); no EXIF date -> left out
  "latitude": 56.1629,
  "longitude": 10.2039,
  "crs": "EPSG:4326",                         # ⚠ "latitude/longitude" only fit a geographic CRS; a projected one (UTM) needs x/y -> fix it to EPSG:4326
  "location_source": "exif_gps",              # NEW: exif_gps | user
  "coordinate_uncertainty_m": 8,              # NEW: EXIF GPSHPositioningError or you; Camtrap DP has the same field
  "tag_list": ["pollinator-survey", "lavender", "sunny"],
  "group_id": 3,                              # ⚠ int only here, s|i for image_id and id -> use one type (string) everywhere
  "group_name": "Aarhus meadow cam 3",

  # ================= single instances (board) =================
  "instances": [
    {
      "id": 0,                                # ⚠ s|i again -> pick one
      "bbox": [1310.6, 959.4, 212.3, 164.8],  # cx, cy, w, h (see bbox_format)
      "conf": 0.91,                           # optional: fits human annotations too (no score)
      "angle": 17.5,                          # our models only give upright boxes (0 / left out); this is an imagined rotated-box model
      "source": "prediction",                 # NEW: prediction | annotation -> same format for training labels
      "label": "insect",                      # NEW: what the detector calls it (insect, arthropod, or the SAM 3 prompt)
      "taxon": "Bombus terrestris",           # NEW: the classifier's answer (board has NO class at all)
      "taxon_conf": 0.83,                     # NEW
      "taxon_rank": "species",                # NEW
      "taxon_options": [                      # NEW: best name per rank (BioCLIP) -> pick a broader one later without re-running
        {"name": "Bombus terrestris", "conf": 0.83, "rank": "species"},
        {"name": "Bombus", "conf": 0.97, "rank": "genus"},
        {"name": "Apidae", "conf": 0.99, "rank": "family"},
        {"name": "Hymenoptera", "conf": 1.0, "rank": "order"}
      ],
      "polygon": [[1406.5, 949.7], [1356.7, 1021.6], [1260.9, 1031.3], [1214.7, 969.1], [1264.5, 897.2], [1360.3, 887.5]],   # NEW: outline, segmentation models only (flat-bug, SAM 3)
      "extra_information": {"visiting_flower": true}   # NEW: per-instance free field (board has it only for the image)
    },
    {
      "id": 1,
      "bbox": [3069.9, 2439.7, 96.4, 58.9],
      "conf": 0.47,
      "angle": 0.0,
      "source": "prediction",
      "label": "insect",
      "taxon": "Syrphidae",                   # species and genus under 0.5 -> answered at family
      "taxon_conf": 0.62,
      "taxon_rank": "family",
      "taxon_options": [
        {"name": "Episyrphus balteatus", "conf": 0.21, "rank": "species"},
        {"name": "Episyrphus", "conf": 0.34, "rank": "genus"},
        {"name": "Syrphidae", "conf": 0.62, "rank": "family"},
        {"name": "Diptera", "conf": 0.97, "rank": "order"}
      ],
      "polygon": [[3024.1, 2415.3], [3117.8, 2412.0], [3116.2, 2466.9], [3022.5, 2468.4]],
      "extra_information": {}
    }
  ],

  # ================= extra (board) =================
  "extra_information": {
    "camera_make": "Raspberry Pi",
    "camera_model": "HQ Camera (IMX477)",
    "camera_serial": "RPI-HQ-0042",
    "exposure_time_s": 0.002,
    "iso": 100,
    "altitude_m": 52.4,
    "file_modified": "2026-07-14T10:32:07+02:00",
    "note": "a suspiciously fluffy bumblebee"
  }
}
```
