"""Minimal web UI (Gradio): image in, detector + classifier, threshold, (text prompt / names), result out.
Started by `python main.py`."""

import base64
import os
import threading
from collections import Counter

from .engine import Zoo
from .registry import CLASSIFIERS, GATED_GUIDE_URL, MODELS, get_classifier, get_model, size_text
from .weights import GatedModelError, hf_token, is_downloaded

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")    # no usage statistics sent to Gradio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = "Detect"
NONE = "none"
PROMPT_LOCKED = "Text prompt (not used by this detector)"
PROMPT_OPEN = "What to look for, e.g. bee  or  bee, butterfly  (empty = %s)"
CLASSES_HINT = "Names to choose from, e.g. Apis mellifera, Bombus terrestris  (empty = arthropod orders)"


def _data_uri(filename, mime):
    """Logos are embedded in the page, so the UI works offline and needs no file serving."""
    path = os.path.join(HERE, "assets", filename)
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))


def header_html():
    insectai = _data_uri("logo_insectai.svg", "image/svg+xml")
    cost = _data_uri("logo_cost_eu.jpg", "image/jpeg")
    return """
<div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap">
  <div style="display:flex; align-items:center; gap:14px">
    <a href="https://insectai.eu/" target="_blank"><img src="{insectai}" alt="InsectAI" style="height:56px"></a>
    <span style="font-size:1.35rem; font-weight:600">Model zoo</span>
  </div>
  <a href="https://www.cost.eu/actions/CA22129/" target="_blank"
     style="background:#fff; border-radius:8px; padding:4px 10px; line-height:0">
    <img src="{cost}" alt="COST - Funded by the European Union" style="height:36px"></a>
</div>""".format(insectai=insectai, cost=cost)


def _choices(cards):
    return [(c.name + ("  (gated)" if c.gated else ""), c.name) for c in cards]


def build(model, device, threshold, iou, output_dir, example_image, prompt=None, classifier="auto", classes=None):
    """Create the Gradio app (without starting it)."""
    import gradio as gr

    zoo = Zoo(device)
    first = get_model(model)
    first_cls = get_classifier(classifier, first)

    def placeholder(card):
        return PROMPT_OPEN % card.default_prompt if card.text_prompt else PROMPT_LOCKED

    def gated_note(*cards):
        """One line under the model lists, only for gated models, with the link to the how-to."""
        notes = []
        for card in cards:
            if card is None or not card.gated:
                continue
            if is_downloaded(card):
                state = "access OK"
            elif hf_token():
                state = "token found, first run downloads %s" % size_text(card.weights.size)
            else:
                state = "needs free access + a Hugging Face token"
            notes.append("🔒 **%s is gated**: %s · [How to get access (5 min)](%s)" % (card.name, state,
                                                                                     GATED_GUIDE_URL))
        return "  \n".join(notes)

    def on_detector_change(name):
        card = get_model(name)
        pair = get_classifier("auto", card)                  # the detector's own classifier, or none
        note = gated_note(card, pair)
        return ((threshold if threshold is not None else card.default_threshold),
                gr.update(interactive=card.text_prompt, placeholder=placeholder(card)),
                gr.update(value=pair.name if pair else NONE),
                gr.update(value=note, visible=bool(note)))

    def on_classifier_change(det_name, cls_name):
        cls = get_classifier(cls_name)
        note = gated_note(get_model(det_name), cls)
        return gr.update(visible=bool(cls and cls.classes)), gr.update(value=note, visible=bool(note))

    def run(det_name, cls_name, thr, text, classes_text, image_path):
        """The Detect button itself shows what is going on (download MB, loading, running) until the result is in."""
        if not image_path:
            raise gr.Error("Add an image first.")
        card, cls = get_model(det_name), get_classifier(cls_name)
        names = [c.strip() for c in (classes_text or "").split(",") if c.strip()] or None
        warnings, result = [], {}
        status = {"text": "Loading %s ... (please wait)" % card.name}

        def log(msg):
            print(msg)
            if msg.startswith("WARNING"):
                warnings.append(msg[len("WARNING: "):])

        def progress_for(c):
            def on_download(done, total, msg):
                status["text"] = ("Downloading %s · %s / %s (please wait)" % (c.name, size_text(done),
                                                                            size_text(total)) if total else msg)
            return on_download

        def work():                      # runs in a thread, so the button can be updated meanwhile
            try:
                zoo.load(card, progress_for(card))
                if cls:
                    status["text"] = "Loading %s ... (please wait)" % cls.name
                zoo.load_classifier(cls, progress_for(cls) if cls else None)
                status["text"] = "Running %s%s ... (please wait)" % (card.name, " + " + cls.name if cls else "")
                folder = card.name + ("+" + cls.name if cls else "")
                result["out"] = zoo.run_file(image_path, thr, iou if iou is not None else card.default_iou,
                                             os.path.join(output_dir, folder), text, names)
            except Exception as e:
                result["error"] = e

        zoo.log = log
        for c in (card, cls):
            if c is not None and not is_downloaded(c):
                print("Downloading %s weights (%s) ..." % (c.name, size_text(c.weights.size)))
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        shown = None
        while worker.is_alive():
            if status["text"] != shown:
                shown = status["text"]
                yield gr.update(), gr.update(value=shown, interactive=False)
            worker.join(0.25)

        ready = gr.update(value=DETECT, interactive=True)
        error = result.get("error")
        if error is not None:
            yield gr.update(), ready
            if isinstance(error, GatedModelError):
                print()
                print(error)
                raise gr.Error("A gated model needs access and your Hugging Face token first. "
                               "Follow the 'How to get access' link under the model lists.")
            raise gr.Error("%s failed: %s" % (card.name, error))
        _, dets, secs, files = result["out"]
        for w in warnings:
            gr.Warning(w)
        taxa = Counter(d.taxon for d in dets if d.taxon)
        found = "%d found" % len(dets) + ("".join(" · %s ×%d" % t for t in taxa.most_common(2)) if taxa else "")
        print("%s: %s, %.2fs on %s -> %s" % (card.name + ("+" + cls.name if cls else ""), found, secs, zoo.device,
                                           os.path.dirname(files[0])))
        yield gr.update(value=files[0], label="%s · %.1f s · %s" % (found, secs, zoo.device)), ready

    with gr.Blocks(title="InsectAI model zoo") as demo:
        gr.HTML(header_html())
        with gr.Row(equal_height=True):
            image_in = gr.Image(type="filepath", label="Image", sources=["upload", "clipboard"], height=440,
                                buttons=["fullscreen"],
                                value=example_image if example_image and os.path.isfile(example_image) else None)
            image_out = gr.Image(label="Result", interactive=False, height=440, buttons=["download", "fullscreen"])
        with gr.Row():
            det_dd = gr.Dropdown(_choices(MODELS.values()), value=first.name, label="Detector", scale=1)
            cls_dd = gr.Dropdown([("none", NONE)] + _choices(CLASSIFIERS.values()),
                                 value=first_cls.name if first_cls else NONE, label="Classifier", scale=1)
            thr = gr.Slider(0.01, 0.99, step=0.01, label="Threshold", scale=2,
                            value=threshold if threshold is not None else first.default_threshold)
        note = gated_note(first, first_cls)
        gated_md = gr.Markdown(note, visible=bool(note))
        text = gr.Textbox(show_label=False, value=prompt or "", max_lines=1, interactive=first.text_prompt,
                          placeholder=placeholder(first))
        classes_box = gr.Textbox(show_label=False, value=classes or "", max_lines=1, placeholder=CLASSES_HINT,
                                 visible=bool(first_cls and first_cls.classes))
        run_btn = gr.Button(DETECT, variant="primary")

        det_dd.change(on_detector_change, det_dd, [thr, text, cls_dd, gated_md])
        cls_dd.change(on_classifier_change, [det_dd, cls_dd], [classes_box, gated_md])
        inputs = [det_dd, cls_dd, thr, text, classes_box, image_in]
        run_btn.click(run, inputs, [image_out, run_btn], show_progress="hidden")
        text.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
        classes_box.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
    return demo


def launch(model, device, threshold, iou, output_dir, example_image, port=None, prompt=None, classifier="auto",
           classes=None):
    import gradio as gr

    demo = build(model, device, threshold, iou, output_dir, example_image, prompt, classifier, classes)
    print("\nOpening the UI in your browser (if it does not open, use the http://127.0.0.1:... address below).")
    print("Results are also saved to %s. Press Ctrl+C here to stop.\n" % output_dir)
    os.makedirs(output_dir, exist_ok=True)
    demo.queue().launch(inbrowser=True, server_name="127.0.0.1", server_port=port, allowed_paths=[output_dir],
                        theme=gr.themes.Soft(primary_hue="green"), footer_links=[],
                        css=".option-list { max-height: 320px !important; }")    # long model lists scroll
