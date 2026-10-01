"""
InsectAI model zoo - find insects in your images (detector) and say what they are (classifier).

    python main.py                                   opens a small web UI in your browser
    python main.py --list_models                     lists the detectors and classifiers
    python main.py --input_image my_photo.jpg        runs the default detector (+ its classifier) on one image
    python main.py -m flatbug-m -c bioclip-2.5 -f my_images    flat-bug finds, BioCLIP 2.5 names, a whole folder
    python main.py --check                           checks your hardware (GPU / RAM) and which models fit
    python main.py --help                            all options

Weights are downloaded from the original authors on first use (see docs/WEIGHTS.md).
"""

# ------------------------------------------------------------------ settings (defaults; command-line arguments override)
MODEL = "insectdct-v8-m"                 # detector, see `python main.py --list_models`
CLASSIFIER = "auto"                      # auto = the detector's own classifier (insectDCT -> insectdct-cls-v7, others
                                         # none), "none", or a classifier name, e.g. "bioclip-2.5"
THRESHOLD = None                         # detection confidence 0-1; None = the model's recommended value
CLS_THRESHOLD = None                     # classification confidence 0-1 (None = the classifier's own choice; BioCLIP
                                         # without names: 0.5). BioCLIP answers at the deepest rank at least this sure:
                                         # lower = more specific (species), higher = surer (genus, family, order)
IOU = None                               # overlap above which two boxes count as the same insect; None = model default
PROMPT = None                            # what text-prompt detectors (sam3) look for, e.g. "bee" or "bee, butterfly"
CLASSES = None                           # names for zero-shot classifiers (bioclip): "Apis mellifera, Bombus terrestris"
                                         # or a .txt file with one name per line; None = every species it knows
BIOCLIP_INSECT_TAXA = True               # BioCLIP without your own names: True = picks from every insect species it
                                         # knows (~250,000); False = the whole tree of life (~800,000: plants, birds...)
INPUT_IMAGE = "images/test_4_domains.jpg"  # one image (a light trap, lab, field + camera-trap mosaic) ...
INPUT_FOLDER = None                      # ... or a folder, e.g. "images" (used instead of INPUT_IMAGE when set)
OUTPUT_DIR = "output"                    # results go to OUTPUT_DIR/<detector>[+<classifier>]/
WEIGHTS_DIR = None                       # where model weights are kept; None = weights/ next to this file. Another
                                         # drive with more room: e.g. "D:/insect-zoo-weights" (or --weights_dir)
DEVICE = "auto"                          # auto = NVIDIA GPU (cuda) -> Apple GPU (mps) -> CPU; or "cpu", "cuda:1", ...
CAMTRAPDP = False                        # True = also write a Camtrap DP data package (the camera-trap data standard,
                                         # e.g. for GBIF) to OUTPUT_DIR/.../camtrap-dp/, same as --camtrapdp
CAMTRAPDP_INFO = dict(                   # what Camtrap DP needs to know; check it before you share a package
    project="Insect camera trap",        #   project title
    contributor="",                      #   your name or organisation (contact); the zoo is listed as well
    deployment_id=None,                  #   camera / site name; None = the images folder's name (or --deployment_id)
    latitude=None, longitude=None,       #   camera position in decimal degrees (WGS84); None = the photos' GPS, if any
    capture_method="timeLapse",          #   "timeLapse" (photos at set times) or "activityDetection" (motion trigger)
    sampling_design="targeted",          #   simpleRandom, systematicRandom, clusteredRandom, experimental, targeted
                                         #   or opportunistic
    timezone=None,                       #   the camera clock's zone, used when a photo does not store it: "+02:00" or
                                         #   "Europe/Copenhagen"; None = this computer's zone
    data_license=None,                   #   licence of the data, e.g. "CC-BY-4.0" or "CC0-1.0" (GBIF needs one)
    media_license=None,                  #   licence of the photos, if you share them
)

# Hugging Face token, only needed for GATED models (sam3); how to get one (5 min): docs/GATED_MODELS.md. Best: run
# `hf auth login` once, or put the token in hf_token.txt next to this file (git never uploads that file). Pasting it
# here also works, but git would upload it with main.py, so the zoo warns you.
HF_TOKEN = ""
# -------------------------------------------------------------------------------------------------------------------

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")      # never let Ultralytics pip-install packages while running
os.environ.setdefault("YOLO_OFFLINE", "true")           # no Ultralytics usage statistics or update checks either
if WEIGHTS_DIR:                                            # read by zoo/weights.py (as INSECT_ZOO_WEIGHTS)
    os.environ["INSECT_ZOO_WEIGHTS"] = os.path.abspath(WEIGHTS_DIR)
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()              # read by zoo/weights.py when downloading gated models
    print("NOTE: your Hugging Face token is written in main.py, which git uploads with the code (also in forks). "
          "Move it: run `hf auth login` (or put it in hf_token.txt, which git ignores), then set HF_TOKEN = \"\" "
          "again. See docs/GATED_MODELS.md.")



def use_own_venv():
    """Started with another Python than the zoo's .venv (e.g. `python main.py` in a new terminal where .venv is not
    activated)? Then run again with the .venv's Python: another Python on the computer can have older packages (an old
    flat-bug there fails with 'Unknown hyperparameter'). INSECT_ZOO_ANY_PYTHON=1 turns this off."""
    venv = os.path.join(HERE, ".venv")
    exe = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    if os.environ.get("INSECT_ZOO_ANY_PYTHON") or not os.path.isfile(exe):
        return
    if os.path.normcase(os.path.realpath(sys.prefix)) == os.path.normcase(os.path.realpath(venv)):
        return
    print("NOTE: started with %s, not with the zoo's own .venv: running with %s instead (activate .venv first to "
          "skip this step)." % (sys.executable, exe), flush=True)
    import subprocess
    proc = subprocess.Popen([exe, os.path.abspath(sys.argv[0])] + sys.argv[1:])
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:           # Ctrl+C reaches the zoo itself too: wait until it has stopped
            continue


if __name__ == "__main__":
    use_own_venv()
if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is needed (you have %s). See README 'Prerequisites'." % sys.version.split()[0])
if os.name == "nt" and len(HERE) > 140:
    print("WARNING: this folder has a very long path (%d characters). Windows cannot load some libraries from deep "
          "paths (error 'The filename or extension is too long'). If that happens, move the folder somewhere short, "
          "e.g. C:\\code\\insect-model-zoo, and create the .venv again.\n" % len(HERE))

sys.path.insert(0, HERE)
from zoo import export, hardware, results                                                # noqa: E402
from zoo.registry import CLASSIFIERS, MODELS, get_classifier, get_model, models_table    # noqa: E402
from zoo import registry                                                                  # noqa: E402

registry.BIOCLIP_INSECT_TAXA = BIOCLIP_INSECT_TAXA
export.TIMEZONE = CAMTRAPDP_INFO.get("timezone")                  # ISIR: capture times in UTC only with a known zone

EXAMPLES = """
examples:
  python main.py                                         open the web UI
  python main.py -i images/test_image.jpg                default detector + its classifier on one image
  python main.py -m flatbug-s -f path/to/images -o results
  python main.py -m flatbug-m -c insectdct-cls-v7        flat-bug finds the insects, insectDCT's classifier names them
  python main.py -m flatbug-m -c bioclip-2.5 --classes "Apis mellifera, Bombus terrestris, Eristalis tenax"
  python main.py -m insectdct-v8-s -c none -t 0.25 -d cpu -i photo.jpg     detection only, on CPU
  python main.py -m sam3 -p "bee, butterfly" -c bioclip-2.5    text prompt (sam3 is GATED: docs/GATED_MODELS.md)
  python main.py --download all                          fetch every model's weights now (e.g. before going offline)
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="python main.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="InsectAI model zoo: a detector finds the insects, an optional classifier says what they are. "
                    "Without arguments a web UI opens in your browser.",
        epilog="models:\n" + models_table() + "\n" + EXAMPLES)
    p.add_argument("-m", "--model", help="detector to use (default: %s). See the list below." % MODEL)
    p.add_argument("-c", "--classifier", help="classifier: auto (default: the detector's own, if any), none, or a name")
    p.add_argument("-t", "--threshold", type=float, help="confidence threshold 0-1 (default: the model's own value)")
    p.add_argument("--iou", type=float, help="overlap (IoU) threshold for merging duplicate boxes (default: model's)")
    p.add_argument("--cls_threshold", type=float, help="classification confidence 0-1: BioCLIP without names answers at "
                                                       "the deepest rank at least this sure (default 0.5; lower = more "
                                                       "specific); other classifiers: below it = Unsure")
    p.add_argument("-p", "--prompt", help='what to look for, for text-prompt detectors (sam3): "bee" or "bee, fly"')
    p.add_argument("--classes", help='names for zero-shot classifiers (bioclip): "Apis mellifera, Bombus terrestris" '
                                     'or a .txt file with one name per line (default: every insect species '
                                     'BioCLIP knows, see BIOCLIP_INSECT_TAXA in main.py)')
    p.add_argument("-i", "--input_image", help="one image to process (default: %s)" % INPUT_IMAGE)
    p.add_argument("-f", "--input_folder", help="process every image in this folder")
    p.add_argument("-o", "--output_dir", help="where results go (default: %s/<detector>+<classifier>)" % OUTPUT_DIR)
    p.add_argument("-d", "--device", help="auto (default), cpu, cuda, cuda:N or mps")
    p.add_argument("-l", "--list_models", "--list", action="store_true", help="list the models and exit")
    p.add_argument("--check", action="store_true", help="show GPU / RAM and which models fit, then exit")
    p.add_argument("--download", metavar="MODEL", help="only download the weights of MODEL ('all' = every model)")
    p.add_argument("--ui", action="store_true", help="open the web UI (the default when no arguments are given)")
    p.add_argument("--camtrapdp", "--camtrapDP", "--camtrap_dp", "--camtrap-dp", action="store_true",
                   help="also write a Camtrap DP data package (camera-trap data standard) to <output>/camtrap-dp/")
    p.add_argument("--latitude", type=float, help="camera position for ISIR and Camtrap DP (decimal degrees; default: photo GPS)")
    p.add_argument("--longitude", type=float, help="camera position for ISIR and Camtrap DP (decimal degrees; default: photo GPS)")
    p.add_argument("--deployment_id", help="Camtrap DP: camera / site name (default: the images folder's name)")
    p.add_argument("--weights_dir", help="where model weights are kept (default: WEIGHTS_DIR in main.py, else weights/)")
    p.add_argument("--port", type=int, help="port for the web UI (default: first free port from 7860)")
    return p


def resolve(path, default):
    """Paths typed on the command line are relative to where you are; defaults are relative to this folder."""
    if path is None:
        return default if default is None or os.path.isabs(default) else os.path.join(HERE, default)
    return os.path.abspath(path)


def check_range(name, value, zero_ok=False):
    if value is not None and not (0 <= value < 1 if zero_ok else 0 < value < 1):
        sys.exit("--%s must be between 0 and 1 (got %s)" % (name, value))
    return value


def parse_classes(value):
    """ "a, b, c" or a text file with one name per line -> list of names (None = the classifier's defaults)."""
    if not value:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        names = [c.strip() for c in value.split(",") if c.strip()]
    return names or None


def output_folder(base, detector, classifier):
    return os.path.join(base, detector.name + ("+" + classifier.name if classifier else ""))


def load_or_explain(zoo, card, classifier=False):
    """Load a model; for a gated model without access, print what to do (and the guide link) instead of a traceback."""
    from zoo.weights import GatedModelError
    try:
        return zoo.load_classifier(card) if classifier else zoo.load(card)
    except GatedModelError as e:
        sys.exit("\n" + str(e) + "\n")
    except RuntimeError as e:                      # weights folder missing, disk full, download failed: say so
        if any(k in str(e) for k in ("weights folder", "disk space", "Could not download")):
            sys.exit("\n" + str(e) + "\n")
        raise


def run_cli(args, card, classifier):
    from zoo.engine import Zoo

    folder = resolve(args.input_folder, INPUT_FOLDER if args.input_image is None else None)
    if folder:
        if not os.path.isdir(folder):
            sys.exit("Input folder not found: %s" % folder)
        images = results.list_images(folder)
        if not images:
            sys.exit("No images (%s) in %s" % (", ".join(results.IMAGE_EXTENSIONS), folder))
    else:
        image = resolve(args.input_image, INPUT_IMAGE)
        if not os.path.isfile(image):
            sys.exit("Input image not found: %s" % image)
        images = [image]

    threshold = check_range("threshold", args.threshold if args.threshold is not None else THRESHOLD)
    iou = check_range("iou", args.iou if args.iou is not None else IOU)
    threshold = card.default_threshold if threshold is None else threshold
    cls_threshold = check_range("cls_threshold",
                                args.cls_threshold if args.cls_threshold is not None else CLS_THRESHOLD, zero_ok=True)
    iou = card.default_iou if iou is None else iou
    out_dir = output_folder(resolve(args.output_dir, OUTPUT_DIR), card, classifier)
    prompt = args.prompt if args.prompt is not None else PROMPT
    if prompt and not card.text_prompt:
        print("NOTE: %s does not use a text prompt, ignoring '%s'." % (card.name, prompt))
        prompt = None
    if card.text_prompt and not prompt:
        prompt = card.default_prompt
        print("NOTE: no --prompt given, looking for '%s'. Example: -p \"bee, butterfly\"" % prompt)
    classes = parse_classes(args.classes if args.classes is not None else CLASSES)
    if classes and not (classifier and classifier.classes):
        print("NOTE: --classes is only used by zero-shot classifiers (bioclip-2.5, bioclip-2); ignoring it.")
        classes = None
    if classes:
        classes, problems = registry.clean_latin_names(classes)          # 'apis' -> 'Apis'
        if problems:
            sys.exit("\n".join(problems))

    zoo = Zoo(args.device or DEVICE)
    for note in zoo.hw.notes:
        print("NOTE: " + note)
    load_or_explain(zoo, card)
    if classifier:
        load_or_explain(zoo, classifier, classifier=True)
    print("Detector %s (threshold %.3f, IoU %.2f%s)" % (card.name, threshold, iou,
                                                       ", prompt '%s'" % prompt if prompt else ""))
    if classifier:
        print("Classifier %s%s" % (classifier.name, (" (your %d name%s + %s)" % (
            len(classes), "" if len(classes) == 1 else "s", registry.bioclip_default_text())) if classes else
            " (no names given: %s)" % registry.bioclip_empty_text() if classifier.classes else ""))
    print("%d image(s)\n" % len(images))

    camtrapdp = args.camtrapdp or CAMTRAPDP
    for name, value, low, high in (("latitude", args.latitude, -90, 90), ("longitude", args.longitude, -180, 180)):
        if value is not None and not low <= value <= high:
            sys.exit("--%s must be between %d and %d (got %s)" % (name, low, high, value))
    lat, lon = (args.latitude if args.latitude is not None else CAMTRAPDP_INFO.get("latitude"),
                args.longitude if args.longitude is not None else CAMTRAPDP_INFO.get("longitude"))
    export.LOCATION = (lat, lon) if lat is not None and lon is not None else None     # ISIR: photos without GPS
    rows, failed, entries, isir_files = [], [], [], []
    for n, path in enumerate(images, 1):
        try:
            image, dets, secs, files = zoo.run_file(path, threshold, iou, out_dir, prompt, classes, cls_threshold)
        except Exception as e:
            print("[%d/%d] %s: FAILED (%s)" % (n, len(images), os.path.basename(path), e))
            failed.append(path)
            continue
        taxa = Counter(d.taxon for d in dets if d.taxon)
        print("[%d/%d] %s: %d detection(s)%s, %.2fs" % (
            n, len(images), os.path.basename(path), len(dets),
            (": " + ", ".join("%s x%d" % t for t in taxa.most_common(5))) if taxa else "", secs))
        rows += [d.row(os.path.basename(path), card.name, classifier.name if classifier else "") for d in dets]
        entries.append(export.Entry(path, os.path.relpath(path, folder) if folder else os.path.basename(path),
                                    image.shape[1], image.shape[0], dets))
        isir_files += [f for f in files if f.endswith("_isir.json")]

    cls_name = classifier.name if classifier else ""
    if len(images) > 1:
        results.write_summary_csv(os.path.join(out_dir, "all_detections.csv"), rows)
        export.write_coco(os.path.join(out_dir, "coco.json"), entries, card.name, cls_name)
        with open(os.path.join(out_dir, "isir.jsonl"), "w", encoding="utf-8") as f:     # one ISIR record per line
            for p in isir_files:
                with open(p, encoding="utf-8") as g:
                    f.write(json.dumps(json.load(g)) + "\n")
    if camtrapdp and entries:
        write_camtrapdp(args, out_dir, entries, card, classifier)
    print("\nDone: %d detection(s) in %d image(s)%s. Results in:\n  %s"
          % (len(rows), len(images) - len(failed), " (%d failed)" % len(failed) if failed else "", out_dir))


def write_camtrapdp(args, out_dir, entries, card, classifier):
    import importlib
    info = dict(CAMTRAPDP_INFO)
    info.update({k: v for k, v in (("deployment_id", args.deployment_id), ("latitude", args.latitude),
                                   ("longitude", args.longitude)) if v is not None})
    # a classifier's own class names -> scientific names (e.g. insectDCT 'Aranaea' -> 'Araneae'), if it has a map
    name_map = getattr(importlib.import_module("zoo.families." + classifier.family), "scientific_name", None) \
        if classifier else None
    folder = os.path.join(out_dir, "camtrap-dp")
    problems = export.write_camtrapdp(folder, entries, card.name, classifier.name if classifier else "", info,
                                      name_map)
    print("\nCamtrap DP data package: %s" % folder)
    print("  project '%s', capture method %s, sampling design %s (change CAMTRAPDP_INFO at the top of main.py if "
          "that is wrong)" % (info["project"], info["capture_method"], info["sampling_design"]))
    for problem in problems:
        print("  NOT VALID YET: " + problem)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.weights_dir:
        from zoo import weights
        weights.WEIGHTS_DIR = os.path.abspath(args.weights_dir)

    if args.list_models:
        print(models_table())
        return
    try:
        card = get_model(args.model or MODEL)
        classifier = get_classifier(args.classifier or CLASSIFIER, card)
    except KeyError as e:
        sys.exit(e.args[0])

    if args.check:
        print(hardware.report(hardware.probe(), list(MODELS.values()) + list(CLASSIFIERS.values())))
        return
    if args.download:
        from zoo.registry import GATED_GUIDE_URL
        from zoo.weights import GatedModelError, ensure_weights
        name = args.download.strip().lower()
        if name == "all":
            cards = list(MODELS.values()) + list(CLASSIFIERS.values())
        else:
            try:
                cards = [MODELS.get(name) or CLASSIFIERS.get(name) or get_model(name)]
            except KeyError as e:
                sys.exit(e.args[0])
        failed = []
        for c in cards:
            try:
                print("%s -> %s" % (c.name, ensure_weights(c)))
            except GatedModelError as e:
                if len(cards) == 1:
                    sys.exit("\n" + str(e) + "\n")
                print("%s -> skipped: GATED, needs access + a Hugging Face token (%s)" % (c.name, GATED_GUIDE_URL))
            except (RuntimeError, OSError) as e:    # download failed, disk full, ...: say so, go on with the next
                if len(cards) == 1:
                    sys.exit("\n" + str(e) + "\n")
                print("%s -> FAILED: %s" % (c.name, str(e).strip().splitlines()[0]))
                failed.append(c.name)
        if failed:
            sys.exit("\nNot downloaded: %s. Run `python main.py --download NAME` for one of them to see the details."
                     % ", ".join(failed))
        return
    cli_only = (args.list_models, args.check, args.download, args.model, args.classifier, args.threshold,
                args.cls_threshold, args.iou, args.prompt, args.classes, args.input_image, args.input_folder,
                args.output_dir, args.device, args.camtrapdp, args.latitude, args.longitude, args.deployment_id)
    given = any(v is not None and v is not False for v in cli_only)     # also counts a value of 0
    if args.ui or not given:              # UI options only (--ui, --port) still mean "open the web UI"
        from zoo.ui import launch
        launch(model=card.name, classifier=args.classifier or CLASSIFIER, device=args.device or DEVICE,
               threshold=check_range("threshold", args.threshold if args.threshold is not None else THRESHOLD),
               iou=check_range("iou", args.iou if args.iou is not None else IOU),
               output_dir=resolve(args.output_dir, OUTPUT_DIR),
               example_image=resolve(None, INPUT_IMAGE), port=args.port,
               prompt=args.prompt if args.prompt is not None else PROMPT,
               classes=", ".join(parse_classes(args.classes if args.classes is not None else CLASSES) or []))
        return
    run_cli(args, card, classifier)


if __name__ == "__main__":
    main()
