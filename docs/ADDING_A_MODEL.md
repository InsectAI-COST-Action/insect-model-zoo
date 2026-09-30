# Adding a model

[← back to the README](../README.md)

1. Add a `ModelCard` to [`zoo/registry.py`](../zoo/registry.py): name, download URL(s), file size, SHA-256, default
   threshold, memory needs, paper DOI, license. Detectors can name a `default_classifier`. Use `kind="classifier"`
   for classifiers, `classes=(...)` for zero-shot ones (unlocks the names field and `--classes`),
   `text_prompt=True` for text-prompted detectors (unlocks the prompt field and `--prompt`), and
   `gated="<Hugging Face page>"` for gated models (token handling and help links come free). Give sizes/versions of
   one model the same `group` and their own `variant` (e.g. flat-bug `N`, `S`, `M`): the UI then lists the family once
   and shows the sizes as buttons, with the largest selected by default. Tags are worked out from these fields.
2. If it is a new kind of model, add `zoo/families/<family>.py` (see the existing families):
   - a detector: class `Model(card, weights_path, device)` with `predict(image_rgb, threshold, iou, prompt=None)`
     returning a list of `Detection`;
   - a classifier: class `Classifier(card, weights_path, device)` with `classify(image_rgb, detections, classes=None)`
     that sets `taxon`, `taxon_score` (and `taxon_rank`) on each detection.
3. Add any new packages to `requirements.txt` and a row to the [Models](../README.md#models) tables.
   If the model has a page in the [InsectAI model database](https://insectai-cost-action.github.io/model-db/), set `model_db="<page>"` (and its datasets in the
   [benchmark database](https://insectai-cost-action.github.io/benchmark-dataset-db/) as `datasets=(("<page>", "<title>"),)`): the UI, `--list_models` and the exports link
   to them. A model that is not in the model database yet is best added there too.
4. Test: `python main.py -m <detector> -c <classifier> -d cpu`, and again on GPU without `-d cpu`.

Suggestions and pull requests are welcome, especially from InsectAI members with models to share.

```text
main.py                 settings at the top, command line, starts the UI
zoo/registry.py         the model list (links, DOIs, licenses, checksums, model-database pages)
zoo/families/           the code that runs each kind of model:
                          detectors: yolo.py (Ultralytics YOLO: ArthroNat; insectdct.py reuses it), yolo_obb.py
                          (oriented boxes: Mothbot), flatbug.py, grounding_dino.py, sam3.py
                          classifiers: insectdct_cls.py, bioclip.py
zoo/weights.py          download + cache + checksum of weights (one download at a time per file)
zoo/hardware.py         device choice and hardware check (--check)
zoo/engine.py           load the detector + classifier and run them (GPU -> CPU fallback)
zoo/results.py          drawing and CSV / COCO JSON output per image
zoo/export.py           COCO JSON and Camtrap DP data package
zoo/ui.py, ui.css,      the web UI (Gradio): layout, and its colours (theme.css)
  theme.css
docs/                   the longer documentation (models, weights, hardware, Camtrap DP, gated models, ...)
images/                 test images: test_4_domains.jpg (light trap, lab, citizen science, camera trap; credits
                        in images/CREDITS.md) and test_image.jpg (a honey bee from our own camera)
assets/                 InsectAI, COST and EU logos for the UI
.github/workflows/      automatic tests on Windows, Linux and macOS
```
