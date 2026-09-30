"""Minimal web UI (Gradio): image in, detector + classifier, threshold, (text prompt / names), result out.
Started by `python main.py`.

Models with several sizes show up once in the lists (e.g. "flat-bug"); picking one shows its size buttons.
"""

import base64
import os
import threading
from collections import Counter

from .engine import Zoo
from .registry import (CLASSIFIERS, GATED_GUIDE_URL, MODELS, SPECIES_TABLE_SURE, bioclip_empty_text,
                       get_classifier, get_model,
                       display_name, download_size, group_default, group_of, groups, clean_latin_names,
                       size_text, tags)
from .weights import GatedModelError, hf_token, is_downloaded

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")    # no usage statistics sent to Gradio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT = "Detect"
STARTING = "Starting ... (please wait)"
NONE = "none"
PROMPT_LOCKED = "Text prompt (not used by this detector)"
PROMPT_OPEN = "What to look for, e.g. bee  or  bee, butterfly  (empty = %s)"
CLASSES_HINT = "Your own Latin names, e.g. Apis mellifera, Bombus terrestris  (empty = %s)"
CSS = """
.option-list { max-height: 320px !important; }                      /* long model lists scroll */
/* the open model list: one box, one row per model with a line between models */
ul.option-list { padding: 6px !important; background: #fff !important; border: 1px solid #e2e8f0 !important;
                 border-radius: 12px !important; box-shadow: 0 12px 28px rgba(15, 23, 42, .12) !important; }
ul.option-list li[role=option] { padding: 10px 12px !important; border-radius: 8px; position: relative; }
ul.option-list li[role=option] + li[role=option]::before {           /* a straight line between two models */
  content: ""; position: absolute; top: 0; left: 10px; right: 10px; border-top: 1px solid #e8edf3; }
ul.option-list li[role=option]:hover, ul.option-list li[role=option].active { background: #f0fdf4 !important; }
ul.option-list li[role=option].selected { background: #dcfce7 !important; font-weight: 600;
                                          box-shadow: inset 3px 0 0 #22c55e; }
ul.option-list li .inner-item { display: none; }                     /* no check mark: the green row shows it */
.zoo-pills { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px; pointer-events: none; }
.zoo-pill { padding: 1px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600; border: 1px solid;
            line-height: 1.5; white-space: nowrap; }
/* name + pills on one line, or wrapped onto more lines when there is no room (the pills keep their shape) */
li[role=option]:has(> .zoo-pills), .secondary-wrap:has(> .zoo-pills) {
  flex-wrap: wrap !important; align-items: center !important; column-gap: 8px; row-gap: 4px;
  white-space: nowrap; word-break: normal !important; }
.secondary-wrap:has(> .zoo-pills) { padding-right: 34px; }            /* leave room for the dropdown arrow */
/* when name + pills do not fit on one line (phone, small window, many tags): name on top, pills centred below */
.zoo-stacked { justify-content: center !important; }
.zoo-stacked > .zoo-pills { flex-basis: 100%; justify-content: center; }
.secondary-wrap.zoo-stacked { padding: 4px 34px 6px; }
.secondary-wrap.zoo-stacked > input { text-align: center; }             /* e.g. 'none' */
.zoo-pill.detector     { color: #15803d; background: #dcfce7; border-color: #86efac; }   /* green  */
.zoo-pill.segmentation { color: #0f766e; background: #ccfbf1; border-color: #5eead4; }   /* teal   */
.zoo-pill.classifier   { color: #6d28d9; background: #ede9fe; border-color: #c4b5fd; }   /* purple */
.zoo-pill.hierarchical { color: #4338ca; background: #e0e7ff; border-color: #a5b4fc; }   /* indigo */
.zoo-pill.zero-shot    { color: #b45309; background: #fef3c7; border-color: #fcd34d; }   /* amber  */
.zoo-pill.text-prompt  { color: #0369a1; background: #e0f2fe; border-color: #7dd3fc; }   /* blue   */
.zoo-pill.gated        { color: #be123c; background: #ffe4e6; border-color: #fda4af; }   /* red    */
/* confidence sliders: an oval (pill) track with a round green handle; the value box on the right updates live */
.zoo-slider input[type="range"] { height: 10px; border-radius: 999px; }
.zoo-slider input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%;
  background: #16a34a; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(15,23,42,.35); }
.zoo-slider input[type="range"]::-moz-range-thumb {
  width: 20px; height: 20px; border: 2px solid #fff; border-radius: 50%; background: #16a34a; }
.zoo-slider input[type="number"] {          /* the live value: a rounded green chip */
  font-weight: 700; color: #15803d; text-align: center; border-radius: 999px !important;
  background: #f0fdf4 !important; border: 1px solid #86efac !important; }
.upload-hint { text-align: center; margin: -2px 0 0; color: #64748b; font-size: .85rem; }
.upload-hint strong { color: #15803d; }
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
  const textWidth = (el, text) => { ruler.font = getComputedStyle(el).font; return ruler.measureText(text).width; };
  const pillsWidth = box => [...box.children].reduce((w, p) => w + p.getBoundingClientRect().width + 4, 0);
  // stacked = name on top, pills centred below. Used when a name + its pills do not fit on one line, and then for
  // every entry of that list (and for both closed fields), so all entries look the same.
  const tooWide = (row, nameWidth, reserve) => {
    const box = row.querySelector(':scope > .zoo-pills');
    return !!box && nameWidth + 8 + pillsWidth(box) > row.clientWidth - reserve;
  };
  const decorate = () => {
    document.querySelectorAll('ul.option-list').forEach(list => {
      const items = [...list.querySelectorAll('li[role=option]')];
      items.forEach(li => put(li, null, li.getAttribute('aria-label')));
      const stacked = items.some(li => tooWide(li, textWidth(li, li.getAttribute('aria-label')), 32));  // 32: padding
      items.forEach(li => li.classList.toggle('zoo-stacked', stacked));
    });
    const fields = [...document.querySelectorAll('input[role=combobox]')];
    fields.forEach(inp => {
      const wrap = inp.parentElement;
      put(wrap, wrap.querySelector('.icon-wrap'), inp.value);
      fit(inp, !!TAGS[inp.value]);
      if (!wrap.dataset.zooClick) {        // a click anywhere in the field still opens the list
        wrap.dataset.zooClick = '1';
        wrap.addEventListener('click', e => { if (e.target !== inp) inp.focus(); });
      }
    });
    const stacked = fields.some(inp => tooWide(inp.parentElement, textWidth(inp, inp.value), 72));  // 72: arrow room
    fields.forEach(inp => inp.parentElement.classList.toggle('zoo-stacked', stacked));
  };
  // The page is always light (the CSS hard-codes light colours): take Gradio's 'dark' body class off.
  // Guarded: classList.remove() of a token that is not there still queues an attribute mutation record
  // once the class attribute exists, and this page observes class changes - an unguarded remove() here
  // would make the observer fire forever, freezing the tab (Firefox and Chromium, dark mode).
  const undark = () => {
    if (document.body.classList.contains('dark'))
      document.body.classList.remove('dark');
  };
  undark();
  new MutationObserver(undark).observe(document.body, {attributes: true, attributeFilter: ['class']});
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


def family_choices(table, none_label=None):
    """One entry per model family (its tags are added as coloured pills by PAGE_JS). `none_label`, when given, adds a
    leading "no model" entry (value NONE), e.g. "none" for the classifier or "whole image" for the detector."""
    head = [(none_label, NONE)] if none_label else []
    return head + [(group, group) for group in groups(table)]


def version_choices(group, table):
    return [("%s · %s" % (c.variant or c.name, size_text(download_size(c))), c.name) for c in groups(table)[group]]


def build(model, device, threshold, iou, output_dir, example_image, prompt=None, classifier="auto", classes=None):
    """Create the Gradio app (without starting it)."""
    import gradio as gr

    zoo = Zoo(device)
    first = get_model(model)
    first_cls = get_classifier(classifier, first)
    for note in zoo.hw.notes:          # e.g. "NVIDIA GPU found but the CPU-only PyTorch is installed"
        print("NOTE: " + note)

    def hardware_notes():               # the same notes as a popup in the browser, once per page load
        for note in zoo.hw.notes:
            gr.Warning(note, duration=30)

    def placeholder(card):
        return PROMPT_OPEN % card.default_prompt if card.text_prompt else PROMPT_LOCKED

    def gated_note(*cards):
        """One line, only for gated models that are not downloaded yet, with the link to the how-to."""
        notes = []
        for card in cards:
            if card is None or not card.gated or is_downloaded(card):
                continue
            state = "token found" if hf_token() else "needs free access + a Hugging Face token"
            notes.append("🔒 **%s is gated**: %s · [How to get access (5 min)](%s)" % (display_name(card), state,
                                                                                     GATED_GUIDE_URL))
        note = "  \n".join(notes)
        return gr.update(value=note, visible=bool(note))

    def cls_slider(card, **kw):
        """Classification confidence: BioCLIP (species table) picks the rank by it, the others say 'Unsure' below it."""
        if card is not None and card.species_table:
            return gr.update(value=SPECIES_TABLE_SURE, info="Lower = more specific (species), higher = surer "
                                                            "(genus, family, order)", **kw)
        return gr.update(value=0.0, info="Names below this show as 'Unsure'", **kw)

    def sizes(group, table, card):
        return gr.update(choices=version_choices(group, table), value=card.name, visible=len(groups(table)[group]) > 1)

    def cls_of(group):
        return None if group == NONE else group_default(group, CLASSIFIERS)

    # handlers read the family dropdowns, not the size buttons (those may still be switching to the new family)
    def on_det_family(group, cls_group):
        # outputs: det_sizes, cls_family, gated_md, thr (the detection-confidence slider), text (the prompt box)
        if group == NONE:                                    # "whole image": no detector, classifier only
            new_cls = cls_group if cls_group != NONE else next(iter(groups(CLASSIFIERS)))   # need a classifier
            return (gr.update(choices=[], value=None, visible=False), gr.update(value=new_cls),
                    gated_note(cls_of(new_cls)), gr.update(visible=False), gr.update(visible=False))
        card = group_default(group, MODELS)
        pair = get_classifier("auto", card)                  # a detector family's own classifier, or none
        return (sizes(group, MODELS, card), gr.update(value=group_of(pair) if pair else NONE),
                gated_note(card, cls_of(cls_group)), gr.update(visible=True), gr.update(visible=True))

    def on_det_version(name):
        if not name:                                         # whole-image mode: no detector picked
            return gr.update(), gr.update()
        card = get_model(name)
        return ((threshold if threshold is not None else card.default_threshold),
                gr.update(interactive=card.text_prompt, placeholder=placeholder(card)))

    def on_cls_family(group, det_group):
        # outputs: cls_sizes, classes_box, gated_md, cls_thr (the classification-confidence slider)
        detector = None if det_group == NONE else group_default(det_group, MODELS)
        if group == NONE:
            return (gr.update(choices=[], value=None, visible=False), gr.update(visible=False),
                    gated_note(detector), gr.update(visible=False))
        card = group_default(group, CLASSIFIERS)
        return (sizes(group, CLASSIFIERS, card), gr.update(visible=bool(card.classes)),
                gated_note(detector, card), cls_slider(card, visible=True))

    def run(det_name, cls_name, det_thr, cls_thr, text, classes_text, image_path):
        """The Detect button itself shows what is going on (download MB, loading, running) until the result is in.
        With no detector (det_name empty) the whole image is one box and only the classifier runs."""
        ready = gr.update(value=DETECT, interactive=True)
        if not image_path:
            yield gr.update(), ready
            raise gr.Error("Add an image first.")
        card = get_model(det_name) if det_name else None
        cls = CLASSIFIERS.get(cls_name) if cls_name else None
        if card is None and cls is None:
            yield gr.update(), ready
            raise gr.Error("Pick a detector, or a classifier to run on the whole image.")
        names = [c.strip() for c in (classes_text or "").split(",") if c.strip()] or None
        problems = []
        if names and cls and cls.classes:
            names, problems = clean_latin_names(names)             # 'apis' -> 'Apis'
        if problems:
            yield gr.update(), ready
            raise gr.Error(" ".join(problems), title="Check the names")
        warnings, result = [], {}
        what = " + ".join([display_name(card) if card else "whole image"] + ([display_name(cls)] if cls else []))
        status = {"text": "Loading %s ... (please wait)" % display_name(card or cls)}

        def log(msg):
            print(msg)
            if msg.startswith("WARNING"):
                warnings.append(msg[len("WARNING: "):])

        def progress_for(c):
            def on_download(done, total, msg):
                status["text"] = ("Downloading %s · %s / %s (please wait)" % (display_name(c), size_text(done),
                                                                            size_text(total)) if total else msg)
            return on_download

        def work():                      # runs in a thread, so the button can be updated meanwhile
            try:
                if card is not None:
                    zoo.load(card, progress_for(card))
                if cls:
                    status["text"] = "Loading %s ... (please wait)" % display_name(cls)
                zoo.load_classifier(cls, progress_for(cls) if cls else None)
                status["text"] = "Running %s ... (please wait)" % what
                if card is not None:
                    folder = card.name + ("+" + cls.name if cls else "")
                    result["out"] = zoo.run_file(image_path, det_thr, iou if iou is not None else card.default_iou,
                                                 os.path.join(output_dir, folder), text, names, cls_threshold=cls_thr)
                else:
                    folder = "whole-image+" + cls.name
                    result["out"] = zoo.run_classify_only(image_path, os.path.join(output_dir, folder), names,
                                                          cls_threshold=cls_thr)
            except Exception as e:
                result["error"] = e

        zoo.log = log
        for c in (card, cls):
            if c is not None and not is_downloaded(c):
                print("Downloading %s weights (%s) ..." % (c.name, size_text(download_size(c))))
        yield gr.update(), gr.update(value=STARTING, interactive=False)    # at once, before any model loads
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        shown = STARTING
        while worker.is_alive():
            if status["text"] != shown:
                shown = status["text"]
                yield gr.update(), gr.update(value=shown, interactive=False)
            worker.join(0.25)

        error = result.get("error")
        if error is not None:
            yield gr.update(), ready
            if isinstance(error, GatedModelError):
                print()
                print(error)
                raise gr.Error("A gated model needs access and your Hugging Face token first. "
                               "Follow the 'How to get access' link under the model lists.")
            raise gr.Error("%s failed: %s" % (what, error))
        _, dets, secs, files = result["out"]
        for w in warnings:
            gr.Warning(w)
        dev = zoo.detector.device or zoo.classifier.device      # no detector in whole-image mode
        taxa = Counter(d.taxon for d in dets if d.taxon)
        if card is not None:
            found = "%d found" % len(dets) + ("".join(" · %s ×%d" % t for t in taxa.most_common(2)) if taxa else "")
        else:                                                  # whole image: report the taxon, not a box count
            found = (dets[0].taxon or "no match") if dets else "no match"
        print("%s: %s, %.2fs on %s -> %s" % (what, found, secs, dev, os.path.dirname(files[0])))
        yield gr.update(value=files[0], label="%s · %.1f s · %s" % (found, secs, dev)), ready

    first_group = group_of(first)
    first_cls_group = group_of(first_cls) if first_cls else NONE
    with gr.Blocks(title="InsectAI model zoo") as demo:
        gr.HTML(header_html())
        example = example_image if example_image and os.path.isfile(example_image) else None
        with gr.Row(equal_height=True):
            with gr.Column():
                image_in = gr.Image(type="filepath", label="Your image", sources=["upload", "clipboard"], height=440,
                                    buttons=["fullscreen"], value=example)
                gr.Markdown(("**Click the image or drag your own photo onto it** to analyse yours (this is just a "
                             "sample)." if example else "**Click or drag a photo here** to get started."),
                            elem_classes="upload-hint")
            with gr.Column():
                image_out = gr.Image(label="Result", interactive=False, height=440,
                                     buttons=["download", "fullscreen"])
        with gr.Row():
            with gr.Column(scale=3, min_width=260), gr.Group():          # one box: list + sizes
                det_family = gr.Dropdown(family_choices(MODELS, none_label="whole image (classifier only)"),
                                         value=first_group, label="Detector")
                det_sizes = gr.Radio(version_choices(first_group, MODELS), value=first.name, show_label=False,
                                     visible=len(groups(MODELS)[first_group]) > 1)
            with gr.Column(scale=3, min_width=260), gr.Group():
                cls_family = gr.Dropdown(family_choices(CLASSIFIERS, none_label="none"), value=first_cls_group,
                                         label="Classifier")
                cls_sizes = gr.Radio(version_choices(first_cls_group, CLASSIFIERS) if first_cls else [],
                                     value=first_cls.name if first_cls else None, show_label=False,
                                     visible=bool(first_cls) and len(groups(CLASSIFIERS)[first_cls_group]) > 1)
            with gr.Column(scale=3, min_width=260):
                thr = gr.Slider(0.01, 0.99, step=0.01, label="Detection confidence", elem_classes="zoo-slider",
                                value=threshold if threshold is not None else first.default_threshold,
                                info="Boxes below this are dropped")
                first_slider = cls_slider(first_cls)
                cls_thr = gr.Slider(0.0, 0.99, step=0.01, label="Classification confidence", elem_classes="zoo-slider",
                                    value=first_slider["value"], info=first_slider["info"], visible=bool(first_cls))
        gated_md = gr.Markdown(visible=False)
        text = gr.Textbox(show_label=False, value=prompt or "", max_lines=1, interactive=first.text_prompt,
                          placeholder=placeholder(first))
        classes_box = gr.Textbox(show_label=False, value=classes or "", max_lines=1,
                                 placeholder=CLASSES_HINT % bioclip_empty_text(),
                                 visible=bool(first_cls and first_cls.classes))
        run_btn = gr.Button(DETECT, variant="primary")

        det_family.change(on_det_family, [det_family, cls_family], [det_sizes, cls_family, gated_md, thr, text])
        det_sizes.change(on_det_version, det_sizes, [thr, text])
        cls_family.change(on_cls_family, [cls_family, det_family], [cls_sizes, classes_box, gated_md, cls_thr])
        inputs = [det_sizes, cls_sizes, thr, cls_thr, text, classes_box, image_in]
        run_btn.click(run, inputs, [image_out, run_btn], show_progress="hidden")
        text.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
        classes_box.submit(run, inputs, [image_out, run_btn], show_progress="hidden")
        demo.load(lambda d, c: gated_note(None if d == NONE else group_default(d, MODELS), cls_of(c)),
                  [det_family, cls_family], gated_md)
        demo.load(hardware_notes)
    return demo


def launch(model, device, threshold, iou, output_dir, example_image, port=None, prompt=None, classifier="auto",
           classes=None):
    import gradio as gr

    demo = build(model, device, threshold, iou, output_dir, example_image, prompt, classifier, classes)
    os.makedirs(output_dir, exist_ok=True)
    _, local_url, _ = demo.queue().launch(
        inbrowser=True, server_name="127.0.0.1", server_port=port, allowed_paths=[output_dir],
        quiet=True, prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="green"), footer_links=[], js=page_js(), css=CSS)
    print("\nThe UI is open in your browser: %s" % local_url)
    print("Results are also saved to %s. Press Ctrl+C here to stop.\n" % output_dir)
    demo.block_thread()
