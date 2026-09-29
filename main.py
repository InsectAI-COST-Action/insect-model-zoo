"""
InsectAI model zoo - run insect detection / segmentation models on your own images.

    python main.py                                   opens a small web UI in your browser
    python main.py --list_models                     lists the models
    python main.py --input_image my_photo.jpg        runs the default model on one image
    python main.py -m flatbug-m -f my_images -t 0.3  runs flat-bug M on every image in a folder
    python main.py --check                           checks your hardware (GPU / RAM) and which models fit
    python main.py --help                            all options

Weights are downloaded from the original authors on first use (see README "How weights are downloaded").
"""

# ------------------------------------------------------------------ settings (defaults; command-line arguments override)
MODEL = "insectdct-v8-m"                 # see `python main.py --list_models`
THRESHOLD = None                         # detection confidence 0-1; None = the model's recommended value
IOU = None                               # overlap above which two boxes count as the same insect; None = model default
PROMPT = None                            # what text-prompt models (sam3) look for, e.g. "bee" or "bee, butterfly"
INPUT_IMAGE = "images/test_image.jpg"    # one image ...
INPUT_FOLDER = None                      # ... or a folder, e.g. "images" (used instead of INPUT_IMAGE when set)
OUTPUT_DIR = "output"                    # results go to OUTPUT_DIR/<model name>/
DEVICE = "auto"                          # auto = NVIDIA GPU (cuda) -> Apple GPU (mps) -> CPU; or "cpu", "cuda:1", ...

# Hugging Face token, only needed for GATED models (sam3). Paste it between the quotes: HF_TOKEN = "hf_..."
# How to get one (5 min): docs/GATED_MODELS.md.  Keep it private: never share or push main.py with your token in it.
HF_TOKEN = ""
# -------------------------------------------------------------------------------------------------------------------

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")      # never let Ultralytics pip-install packages while running
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()              # read by zoo/weights.py when downloading gated models

if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is needed (you have %s). See README 'Prerequisites'." % sys.version.split()[0])
if os.name == "nt" and len(HERE) > 140:
    print("WARNING: this folder has a very long path (%d characters). Windows cannot load some libraries from deep "
          "paths (error 'The filename or extension is too long'). If that happens, move the folder somewhere short, "
          "e.g. C:\\code\\Insect_model_zoo, and create the .venv again.\n" % len(HERE))

sys.path.insert(0, HERE)
from zoo import hardware, results                          # noqa: E402
from zoo.registry import MODELS, get_model, models_table   # noqa: E402

EXAMPLES = """
examples:
  python main.py                                         open the web UI
  python main.py -i images/test_image.jpg                default model on one image
  python main.py -m flatbug-s -f path/to/images -o results
  python main.py -m insectdct-v8-s -t 0.25 -d cpu -i photo.jpg
  python main.py -m sam3 -p "bee, butterfly" -i photo.jpg  text prompt (sam3 is GATED: docs/GATED_MODELS.md)
  python main.py --download all                          fetch every model's weights now (e.g. before going offline)
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="python main.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="InsectAI model zoo: run insect detection / segmentation models on images. "
                    "Without arguments a web UI opens in your browser.",
        epilog="models:\n" + models_table() + "\n" + EXAMPLES)
    p.add_argument("-m", "--model", help="model to use (default: %s). See the list below." % MODEL)
    p.add_argument("-t", "--threshold", type=float, help="confidence threshold 0-1 (default: the model's own value)")
    p.add_argument("--iou", type=float, help="overlap (IoU) threshold for merging duplicate boxes (default: model's)")
    p.add_argument("-p", "--prompt", help='what to look for, for text-prompt models (sam3): "bee" or "bee, butterfly"')
    p.add_argument("-i", "--input_image", help="one image to process (default: %s)" % INPUT_IMAGE)
    p.add_argument("-f", "--input_folder", help="process every image in this folder")
    p.add_argument("-o", "--output_dir", help="where results go (default: %s/<model>)" % OUTPUT_DIR)
    p.add_argument("-d", "--device", help="auto (default), cpu, cuda, cuda:N or mps")
    p.add_argument("-l", "--list_models", "--list", action="store_true", help="list the models and exit")
    p.add_argument("--check", action="store_true", help="show GPU / RAM and which models fit, then exit")
    p.add_argument("--download", metavar="MODEL", help="only download the weights of MODEL ('all' = every model)")
    p.add_argument("--ui", action="store_true", help="open the web UI (the default when no arguments are given)")
    p.add_argument("--port", type=int, help="port for the web UI (default: first free port from 7860)")
    return p


def resolve(path, default):
    """Paths typed on the command line are relative to where you are; defaults are relative to this folder."""
    if path is None:
        return default if default is None or os.path.isabs(default) else os.path.join(HERE, default)
    return os.path.abspath(path)


def check_range(name, value):
    if value is not None and not 0 < value < 1:
        sys.exit("--%s must be between 0 and 1 (got %s)" % (name, value))
    return value


def load_or_explain(zoo, card):
    """Load the model; for a gated model without access, print what to do (and the guide link) instead of a traceback."""
    from zoo.weights import GatedModelError
    try:
        zoo.load(card)
    except GatedModelError as e:
        sys.exit("\n" + str(e) + "\n")


def run_cli(args, card):
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
    iou = card.default_iou if iou is None else iou
    out_dir = os.path.join(resolve(args.output_dir, OUTPUT_DIR), card.name)
    prompt = args.prompt if args.prompt is not None else PROMPT
    if prompt and not card.text_prompt:
        print("NOTE: %s does not use a text prompt, ignoring '%s'." % (card.name, prompt))
        prompt = None
    if card.text_prompt and not prompt:
        prompt = card.default_prompt
        print("NOTE: no --prompt given, looking for '%s'. Example: -p \"bee, butterfly\"" % prompt)

    zoo = Zoo(args.device or DEVICE)
    for note in zoo.hw.notes:
        print("NOTE: " + note)
    load_or_explain(zoo, card)
    print("Model %s, threshold %.3f, IoU %.2f%s, %d image(s)\n"
          % (card.name, threshold, iou, ", prompt '%s'" % prompt if prompt else "", len(images)))

    rows, failed = [], []
    for n, path in enumerate(images, 1):
        try:
            _, dets, secs, _ = zoo.run_file(path, threshold, iou, out_dir, prompt)
        except Exception as e:
            print("[%d/%d] %s: FAILED (%s)" % (n, len(images), os.path.basename(path), e))
            failed.append(path)
            continue
        print("[%d/%d] %s: %d detection(s), %.2fs" % (n, len(images), os.path.basename(path), len(dets), secs))
        rows += [d.row(os.path.basename(path), card.name) for d in dets]

    if len(images) > 1:
        results.write_summary_csv(os.path.join(out_dir, "all_detections.csv"), rows)
    print("\nDone: %d detection(s) in %d image(s)%s. Results in:\n  %s"
          % (len(rows), len(images) - len(failed), " (%d failed)" % len(failed) if failed else "", out_dir))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    if args.list_models:
        print(models_table())
        return
    try:
        card = get_model(args.model or MODEL)
    except KeyError as e:
        sys.exit(e.args[0])

    if args.check:
        print(hardware.report(hardware.probe(), list(MODELS.values())))
        return
    if args.download:
        from zoo.registry import GATED_GUIDE_URL
        from zoo.weights import GatedModelError, ensure_weights
        try:
            cards = list(MODELS.values()) if args.download.lower() == "all" else [get_model(args.download)]
        except KeyError as e:
            sys.exit(e.args[0])
        for c in cards:
            try:
                print("%s -> %s" % (c.name, ensure_weights(c)))
            except GatedModelError as e:
                if len(cards) == 1:
                    sys.exit("\n" + str(e) + "\n")
                print("%s -> skipped: GATED, needs access + a Hugging Face token (%s)" % (c.name, GATED_GUIDE_URL))
        return
    if not argv or args.ui:
        from zoo.ui import launch
        launch(model=card.name, device=args.device or DEVICE,
               threshold=check_range("threshold", args.threshold if args.threshold is not None else THRESHOLD),
               iou=check_range("iou", args.iou if args.iou is not None else IOU),
               output_dir=resolve(args.output_dir, OUTPUT_DIR),
               example_image=resolve(None, INPUT_IMAGE), port=args.port,
               prompt=args.prompt if args.prompt is not None else PROMPT)
        return
    run_cli(args, card)


if __name__ == "__main__":
    main()
