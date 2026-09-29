"""Minimal web UI (Gradio): image in, detector + classifier, threshold, (text prompt / names), result out.
Started by `python main.py`.

Models with several sizes show up once in the lists (e.g. "flat-bug"); picking one shows its size buttons.
"""

import base64
import os
import threading
from collections import Counter

from .engine import Zoo
from .registry import (CLASSIFIERS, GATED_GUIDE_URL, MODELS, bioclip_default_text, get_classifier, get_model,
                       group_default, group_of, groups, latin_name_problems, size_text, tags)
from .weights import GatedModelError, hf_token, is_downloaded

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")    # no usage statistics sent to Gradio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = "Detect"
NONE = "none"
PROMPT_LOCKED = "Text prompt (not used by this detector)"
PROMPT_OPEN = "What to look for, e.g. bee  or  bee, butterfly  (empty = %s)"
CLASSES_HINT = "Latin names, e.g. Apis mellifera, Bombus terrestris  (always compared with %s)"
CSS = """
.option-list { max-height: 320px !important; }                      /* long model lists scroll */
.zoo-pills { display: inline-flex; gap: 4px; margin-left: 8px; flex-shrink: 0; pointer-events: none; }
.secondary-wrap > .zoo-pills { margin-right: 26px; }                 /* leave room for the dropdown arrow */
.zoo-pill { padding: 1px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600; border: 1px solid;
            line-height: 1.5; white-space: nowrap; }
.zoo-pill.detector     { color: #15803d; background: #dcfce7; border-color: #86efac; }   /* green  */
.zoo-pill.segmentation { color: #0f766e; background: #ccfbf1; border-color: #5eead4; }   /* teal   */
.zoo-pill.classifier   { color: #6d28d9; background: #ede9fe; border-color: #c4b5fd; }   /* purple */
.zoo-pill.hierarchical { color: #4338ca; background: #e0e7ff; border-color: #a5b4fc; }   /* indigo */
.zoo-pill.zero-shot    { color: #b45309; background: #fef3c7; border-color: #fcd34d; }   /* amber  */
.zoo-pill.text-prompt  { color: #0369a1; background: #e0f2fe; border-color: #7dd3fc; }   /* blue   */
.zoo-pill.gated        { color: #be123c; background: #ffe4e6; border-color: #fda4af; }   /* red    */
"""

# Runs in the browser. Gradio dropdowns only hold plain text, so the tags are added as coloured pills next to each
# model name: in the open list and in the closed field. Also keeps the page light (Gradio follows the computer's
# dark mode by adding a "dark" class to the page).
PAGE_JS = """() => {
  const TAGS = __TAGS__;
  const html = tags => tags.map(t => '<span class="zoo-pill ' + t.replace(' ', '-') + '">' + t + '</span>').join('');
  const put = (parent, before, name) => {
    let box = parent.querySelector(':scope > .zoo-pills');
    if (!TAGS[name]) { if (box) box.remove(); return; }
    if (!box) { box = document.createElement('span'); box.className = 'zoo-pills'; parent.insertBefore(box, before); }
    if (box.dataset.name !== name) { box.innerHTML = html(TAGS[name]); box.dataset.name = name; }
  };
  const ruler = document.createElement('canvas').getContext('2d');
  const fit = (inp, tagged) => {           // closed field: text box only as wide as the name, pills right after it
    if (!tagged || document.activeElement === inp) { inp.style.flex = ''; inp.style.width = ''; return; }
    ruler.font = getComputedStyle(inp).font;
    inp.style.flex = '0 0 auto';
    inp.style.width = Math.ceil(ruler.measureText(inp.value).width + 4) + 'px';
  };
  const decorate = () => {
    document.body.classList.remove('dark');
    document.querySelectorAll('li[role=option]').forEach(li => put(li, null, li.getAttribute('aria-label')));
    document.querySelectorAll('input[role=combobox]').forEach(inp => {
      const wrap = inp.parentElement;
      put(wrap, wrap.querySelector('.icon-wrap'), inp.value);
      fit(inp, !!TAGS[inp.value]);
      if (!wrap.dataset.zooClick) {        // a click anywhere in the field still opens the list
        wrap.dataset.zooClick = '1';
        wrap.addEventListener('click', e => { if (e.target !== inp) inp.focus(); });
      }
    });
  };
  new MutationObserver(decorate).observe(document.body, {childList: true, subtree: true, attributes: true,
                                                         attributeFilter: ['class']});
  setInterval(decorate, 300);            // the selected value changes without a DOM change
  decorate();
}"""


def page_js():
    import json
    names = {group: tags(cards[0]) for table in (MODELS, CLASSIFIERS) for group, cards in groups(table).items()}
    return PAGE_JS.replace("__TAGS__", json.dumps(names))


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


def family_choices(table, with_none=False):
    """One entry per model family (its tags are added as coloured pills by PAGE_JS)."""
    return ([("none", NONE)] if with_none else []) + [(group, group) for group in groups(table)]


def version_choices(group, table):
    return [("%s · %s" % (c.variant or c.name, size_text(c.weights.size)), c.name) for c in groups(table)[group]]


def build(model, device, threshold, iou, output_dir, example_image, prompt=None, classifier="auto", classes=None):
    """Create the Gradio app (without starting it)."""
    import gradio as gr

    zoo = Zoo(device)
    first = get_model(model)
    first_cls = get_classifier(classifier, first)

    def placeholder(card):
        return PROMPT_OPEN % card.default_prompt if card.text_prompt else PROMPT_LOCKED

    def gated_note(*cards):
        """One line, only for gated models that are not downloaded yet, with the link to the how-to."""
        notes = []
        for card in cards:
            if card is None or not card.gated or is_downloaded(card):
                continue
            state = "token found" if hf_token() else "needs free access + a Hugging Face token"
            notes.append("🔒 **%s is gated**: %s · [How to get access (5 min)](%s)" % (card.name, state,
                                                                                     GATED_GUIDE_URL))
        note = "  \n".join(notes)
        return gr.update(value=note, visible=bool(note))

    def sizes(group, table, card):
        return gr.update(choices=version_choices(group, table), value=card.name, visible=len(groups(table)[group]) > 1)

    def cls_of(group):
        return None if group == NONE else group_default(group, CLASSIFIERS)

    # handlers read the family dropdowns, not the size buttons (those may still be switching to the new family)
    def on_det_family(group, cls_group):
        card = group_default(group, MODELS)
        pair = get_classifier("auto", card)                  # a detector family's own classifier, or none
        return (sizes(group, MODELS, card), gr.update(value=group_of(pair) if pair else NONE),
                gated_note(card, cls_of(cls_group)))

    def on_det_version(name):
        card = get_model(name)
        return ((threshold if threshold is not None else card.default_threshold),
                gr.update(interactive=card.text_prompt, placeholder=placeholder(card)))

    def on_cls_family(group, det_group):
        detector = group_default(det_group, MODELS)
        if group == NONE:
            return (gr.update(choices=[], value=None, visible=False), gr.update(visible=False),
                    gated_note(detector))
        card = group_default(group, CLASSIFIERS)
        return (sizes(group, CLASSIFIERS, card), gr.update(visible=bool(card.classes)),
                gated_note(detector, card))

    def run(det_name, cls_name, thr, text, classes_text, image_path):
        """The Detect button itself shows what is going on (download MB, loading, running) until the result is in."""
        if not image_path:
            raise gr.Error("Add an image first.")
        card, cls = get_model(det_name), (CLASSIFIERS.get(cls_name) if cls_name else None)
        names = [c.strip() for c in (classes_text or "").split(",") if c.strip()] or None
        problems = latin_name_problems(names) if names and cls and cls.classes else []
        if problems:
            raise gr.Error(" ".join(problems), title="Check the names")
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

    first_group = group_of(first)
    first_cls_group = group_of(first_cls) if first_cls else NONE
    with gr.Blocks(title="InsectAI model zoo") as demo:
        gr.HTML(header_html())
        with gr.Row(equal_height=True):
            image_in = gr.Image(type="filepath", label="Image", sources=["upload", "clipboard"], height=440,
                                buttons=["fullscreen"],
                                value=example_image if example_image and os.path.isfile(example_image) else None)
            image_out = gr.Image(label="Result", interactive=False, height=440, buttons=["download", "fullscreen"])
        with gr.Row():
            with gr.Column(scale=3, min_width=260), gr.Group():          # one box: list + sizes
                det_family = gr.Dropdown(family_choices(MODELS), value=first_group, label="Detector")
                det_sizes = gr.Radio(version_choices(first_group, MODELS), value=first.name, show_label=False,
                                     visible=len(groups(MODELS)[first_group]) > 1)
            with gr.Column(scale=3, min_width=260), gr.Group():
                cls_family = gr.Dropdown(family_choices(CLASSIFIERS, with_none=True), value=first_cls_group,
                                         label="Classifier")
                cls_sizes = gr.Radio(version_choices(first_cls_group, CLASSIFIERS) if first_cls else [],
                                     value=first_cls.name if first_cls else None, show_label=False,
                                     visible=bool(first_cls) and len(groups(CLASSIFIERS)[first_cls_group]) > 1)
            with gr.Column(scale=2, min_width=220):
                thr = gr.Slider(0.01, 0.99, step=0.01, label="Threshold",
                                value=threshold if threshold is not None else first.default_threshold)
        gated_md = gr.Markdown(visible=False)
        text = gr.Textbox(show_label=False, value=prompt or "", max_lines=1, interactive=first.text_prompt,
                          placeholder=placeholder(first))
        classes_box = gr.Textbox(show_label=False, value=classes or "", max_lines=1, placeholder=CLASSES_HINT % bioclip_default_text(),
                                 visible=bool(first_cls and first_cls.classes))
        run_btn = gr.Button(DETECT, variant="primary")

        det_family.change(on_det_family, [det_family, cls_family], [det_sizes, cls_family, gated_md])
        det_sizes.change(on_det_version, det_sizes, [thr, text])
        cls_family.change(on_cls_family, [cls_family, det_family], [cls_sizes, classes_box, gated_md])
        inputs = [det_sizes, cls_sizes, thr, text, classes_box, image_in]
        run_btn.click(run, inputs, [image_out, run_btn], show_progress="hidden")
        text.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
        classes_box.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
        demo.load(lambda d, c: gated_note(group_default(d, MODELS), cls_of(c)), [det_family, cls_family], gated_md)
    return demo


def launch(model, device, threshold, iou, output_dir, example_image, port=None, prompt=None, classifier="auto",
           classes=None):
    import gradio as gr

    demo = build(model, device, threshold, iou, output_dir, example_image, prompt, classifier, classes)
    print("\nOpening the UI in your browser (if it does not open, use the http://127.0.0.1:... address below).")
    print("Results are also saved to %s. Press Ctrl+C here to stop.\n" % output_dir)
    os.makedirs(output_dir, exist_ok=True)
    demo.queue().launch(inbrowser=True, server_name="127.0.0.1", server_port=port, allowed_paths=[output_dir],
                        theme=gr.themes.Soft(primary_hue="green"), footer_links=[], js=page_js(),
                        css=CSS)
