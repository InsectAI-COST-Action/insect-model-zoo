"""Minimal web UI (Gradio): image in, model, threshold, (text prompt), result out. Started by `python main.py`."""

import base64
import os
import threading

from .engine import Zoo
from .registry import GATED_GUIDE_URL, MODELS, get_model, size_text
from .weights import GatedModelError, hf_token, is_downloaded

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")    # no usage statistics sent to Gradio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = "Detect"
PROMPT_LOCKED = "Text prompt (not used by this model)"
PROMPT_OPEN = "What to look for, e.g. bee  or  bee, butterfly  (empty = %s)"


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


def build(model, device, threshold, iou, output_dir, example_image, prompt=None):
    """Create the Gradio app (without starting it)."""
    import gradio as gr

    zoo = Zoo(device)
    first = get_model(model)

    def placeholder(card):
        return PROMPT_OPEN % card.default_prompt if card.text_prompt else PROMPT_LOCKED

    def gated_note(card):
        """One line under the model list, only for gated models, with the link to the how-to."""
        if not card.gated:
            return ""
        if is_downloaded(card):
            state = "access OK"
        elif hf_token():
            state = "token found, first run downloads %s" % size_text(card.weights.size)
        else:
            state = "needs free access + a Hugging Face token"
        return "🔒 **%s is gated**: %s · [How to get access (5 min)](%s)" % (card.name, state, GATED_GUIDE_URL)

    def on_model_change(name):
        card = get_model(name)
        note = gated_note(card)
        return ((threshold if threshold is not None else card.default_threshold),
                gr.update(interactive=card.text_prompt, placeholder=placeholder(card)),
                gr.update(value=note, visible=bool(note)))

    def run(name, thr, text, image_path):
        """The Detect button itself shows what is going on (download MB, loading, running) until the result is in."""
        if not image_path:
            raise gr.Error("Add an image first.")
        card = get_model(name)
        warnings, result = [], {}
        status = {"text": "Loading %s ... (please wait)" % card.name}

        def log(msg):
            print(msg)
            if msg.startswith("WARNING"):
                warnings.append(msg[len("WARNING: "):])

        def on_download(done, total, _msg):
            status["text"] = "Downloading %s · %s / %s (please wait)" % (card.name, size_text(done), size_text(total))

        def work():                      # runs in a thread, so the button can be updated meanwhile
            try:
                zoo.load(card, on_download)
                status["text"] = "Running %s ... (please wait)" % card.name
                result["out"] = zoo.run_file(image_path, thr, iou if iou is not None else card.default_iou,
                                             os.path.join(output_dir, card.name), text)
            except Exception as e:
                result["error"] = e

        zoo.log = log
        if not is_downloaded(card):
            print("Downloading %s weights (%s) ..." % (card.name, size_text(card.weights.size)))
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
                raise gr.Error("%s is gated: request access and add your Hugging Face token first. "
                               "Follow the 'How to get access' link under the model list." % card.name)
            raise gr.Error("%s failed: %s" % (card.name, error))
        _, dets, secs, files = result["out"]
        for w in warnings:
            gr.Warning(w)
        print("%s: %d detection(s), %.2fs on %s -> %s" % (card.name, len(dets), secs, zoo.device,
                                                        os.path.dirname(files[0])))
        yield gr.update(value=files[0], label="%d found · %.1f s · %s" % (len(dets), secs, zoo.device)), ready

    with gr.Blocks(title="InsectAI model zoo") as demo:
        gr.HTML(header_html())
        with gr.Row(equal_height=True):
            image_in = gr.Image(type="filepath", label="Image", sources=["upload", "clipboard"], height=440,
                                buttons=["fullscreen"],
                                value=example_image if example_image and os.path.isfile(example_image) else None)
            image_out = gr.Image(label="Result", interactive=False, height=440, buttons=["download", "fullscreen"])
        with gr.Row():
            model_dd = gr.Dropdown([(c.name + ("  (gated)" if c.gated else ""), c.name) for c in MODELS.values()],
                                   value=first.name, label="Model", scale=1)
            thr = gr.Slider(0.01, 0.99, step=0.01, label="Threshold", scale=2,
                            value=threshold if threshold is not None else first.default_threshold)
        gated_md = gr.Markdown(gated_note(first), visible=bool(first.gated))
        text = gr.Textbox(show_label=False, value=prompt or "", max_lines=1, interactive=first.text_prompt,
                          placeholder=placeholder(first))
        run_btn = gr.Button(DETECT, variant="primary")

        model_dd.change(on_model_change, model_dd, [thr, text, gated_md])
        run_btn.click(run, [model_dd, thr, text, image_in], [image_out, run_btn], show_progress="hidden")
        text.submit(run, [model_dd, thr, text, image_in], [image_out, run_btn], show_progress="hidden")
    return demo


def launch(model, device, threshold, iou, output_dir, example_image, port=None, prompt=None):
    import gradio as gr

    demo = build(model, device, threshold, iou, output_dir, example_image, prompt)
    print("\nOpening the UI in your browser (if it does not open, use the http://127.0.0.1:... address below).")
    print("Results are also saved to %s. Press Ctrl+C here to stop.\n" % output_dir)
    os.makedirs(output_dir, exist_ok=True)
    demo.queue().launch(inbrowser=True, server_name="127.0.0.1", server_port=port, allowed_paths=[output_dir],
                        theme=gr.themes.Soft(primary_hue="green"), footer_links=[],
                        css=".option-list { max-height: 320px !important; }")    # long model list scrolls
