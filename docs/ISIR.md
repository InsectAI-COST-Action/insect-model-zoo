# ISIR output

[← back to the README](../README.md)

For every image the zoo writes `<image>_isir.json`: one **ISIR** record (InsectAI Standard Intermediate
Representation, version 1), the shared detection format of the
[InsectAI model database](https://insectai-cost-action.github.io/model-db/). A folder run also writes `isir.jsonl`,
one record per line. The format itself is defined in model-db:
[`static/formats/detection/isir.json`](https://github.com/InsectAI-COST-Action/model-db/blob/main/static/formats/detection/isir.json).

The tests check every record with **model-db's own checker** (pinned to model-db commit `c05b22d`), and check that the
boxes are the same as model-db's own conversion of the zoo's COCO file.

## How the zoo fills it

| ISIR key | What the zoo writes |
|---|---|
| `ir_name`, `ir_id` | `"ISIR"`, `1` |
| `image.id`, `image.file_name` | the file name (relative to the input folder in a folder run) |
| `image.width`, `image.height`, `image.format` | pixels of the image as read (after EXIF rotation), MIME type e.g. `image/jpeg` |
| `instances[].id` | 0, 1, 2, ... within the image |
| `instances[].bbox` | `[centre x, centre y, width, height]` in pixels, **origin bottom-left** (ISIR's convention; COCO and the CSV use top-left) |
| `instances[].confidence` | the detector's confidence; left out in whole-image mode (no detector) |
| `instances[].category_id` | the detector's label: `insect` for all detectors for now, or the prompt word for text-prompt detectors |
| `instances[].polygons` | outlines of segmentation models (flat-bug, SAM 3), and the 4 corners of oriented boxes, same coordinates as `bbox` |
| `model.name` | the detector, e.g. `insectdct-v8-s`; left out in whole-image mode |
| `inference.timestamp` | when the zoo ran, UTC |
| `inference.config` | the settings used: `threshold`, `iou`, `prompt`, `classes`, `cls_threshold` |
| `context.timestamp` | when the photo was taken, in UTC, **only when its time zone is known**: stored in the photo, or `timezone` set in `CAMTRAPDP_INFO` in `main.py` |
| `context.latitude`, `context.longitude`, `context.crs` | the photo's GPS, else `--latitude` / `--longitude`; `EPSG:4326` |
| `extra_information.insect_model_zoo` | everything else (below) |

Not filled: `model.uuid` (model-db has no model uuids yet), `angle` (see oriented boxes), `context.tag_list`,
`group_name`, `group_id`.

### The classifier's answers

ISIR has no field for a taxon, so, as agreed in
[issue #1](https://github.com/InsectAI-COST-Action/insect-model-zoo/issues/1), they go in `extra_information`, linked to
the box by its instance `id`:

```json
"extra_information": {"insect_model_zoo": {
  "classifier": "insectdct-cls-v7",
  "classifications": [{"instance_id": 0, "taxon": "Apis mellifera", "rank": "species", "score": 0.9905}]
}}
```

BioCLIP without your own names also gives `options`: its best name at each rank (species → genus → family → order),
as a fallback list.

### Oriented boxes (Mothbot)

ISIR's `angle` is optional, and model-db's checker does not accept a non-zero angle yet (its rotation convention is
still open). So for an oriented box the zoo writes the upright box around it as `bbox`, the 4 corners as a polygon,
and the angle in the instance's `extra_information` (`angle_deg`: the long side against the image x-axis, degrees
counter-clockwise). See [`oriented_box.json`](isir/oriented_box.json).

### Other fields in `extra_information.insect_model_zoo`

- `capture_time_source`: where `context.timestamp` came from.
- `capture_time_local_unresolved`: the photo's local time when its time zone is unknown (then there is no
  `context.timestamp`; set `timezone` in `CAMTRAPDP_INFO` to get one).
- `file_modified`: the file's time when the photo has no EXIF date. This is *not* the capture time.
- `location_source`: photo GPS or the settings.
- `device`: where the models ran, and `model_db`: the models' pages in the model database.

## Examples

All made by the zoo and checked with model-db's checker:

| File | What it shows |
|---|---|
| [`populated.json`](isir/populated.json) | detector + classifier, capture time (camera zone from the settings) and camera position |
| [`empty.json`](isir/empty.json) | an image where nothing was found |
| [`two_images.jsonl`](isir/two_images.jsonl) | a folder run: one record per line |
| [`oriented_box.json`](isir/oriented_box.json) | Mothbot's oriented boxes (shortened to 2 instances) |

## Later

model-db is adding a converter API
([model-db PR #61](https://github.com/InsectAI-COST-Action/model-db/pull/61)). Once it is merged, the zoo can use it to
write ISIR (and the other formats) instead of its own code; the output should stay the same.
