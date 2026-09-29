"""Minimal web UI (Gradio): image in, detector + classifier, threshold, (text prompt / names), result out.
Started by `python main.py`.

Models with several sizes show up once in the lists (e.g. "flat-bug"); picking one shows its size buttons.
"""

import atexit
import base64
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
from collections import Counter

from .engine import Zoo
from .registry import (CLASSIFIERS, GATED_GUIDE_URL, MODELS, bioclip_default_text, get_classifier, get_model,
                       display_name, group_default, group_of, groups, clean_latin_names, size_text, tags)
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
    document.body.classList.remove('dark');
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
        problems = []
        if names and cls and cls.classes:
            names, problems = clean_latin_names(names)             # 'apis' -> 'Apis'
        if problems:
            raise gr.Error(" ".join(problems), title="Check the names")
        warnings, result = [], {}
        status = {"text": "Loading %s ... (please wait)" % display_name(card)}

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
                zoo.load(card, progress_for(card))
                if cls:
                    status["text"] = "Loading %s ... (please wait)" % display_name(cls)
                zoo.load_classifier(cls, progress_for(cls) if cls else None)
                status["text"] = "Running %s%s ... (please wait)" % (display_name(card),
                                                                    " + " + display_name(cls) if cls else "")
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
            raise gr.Error("%s failed: %s" % (display_name(card), error))
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
        demo.load(hardware_notes)
    return demo


def qr_lines(url):
    """The URL as a QR code for the terminal: black on white (scans in light and dark terminals), two QR rows per
    text line using half blocks. None if the qrcode package is missing."""
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    m = qr.get_matrix()                          # True = dark module, border included
    if len(m) % 2:
        m.append([False] * len(m[0]))
    chars = {(False, False): " ", (True, True): "█", (True, False): "▀", (False, True): "▄"}
    return ["\033[30;47m" + "".join(chars[(m[r][c], m[r + 1][c])] for c in range(len(m[0]))) + "\033[0m"
            for r in range(0, len(m), 2)]


def print_qr(url):
    lines = qr_lines(url)
    if not lines:
        print("  (pip install qrcode to also get a QR code here)")
        return
    try:
        import colorama                          # lets older Windows consoles show the black/white colours
        colorama.just_fix_windows_console()
    except Exception:
        pass
    try:
        print("  Scan to open it on a phone:\n")
        for line in lines:
            print("  " + line)
        print()
    except UnicodeEncodeError:                   # a console that cannot show block characters: skip the QR code
        print("  (this terminal cannot show the QR code)")


def lan_address():
    """This computer's address on the local network (e.g. 192.168.1.23), or None without a network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))               # UDP: nothing is sent, it only picks the network card in use
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


CLOUDFLARED_PATHS = [r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                     r"C:\Program Files\cloudflared\cloudflared.exe"]


def find_cloudflared():
    """Cloudflare's tunnel program, if installed (winget / brew / apt put it on PATH; winget's folder is also checked
    because PATH only updates in a new terminal)."""
    return shutil.which("cloudflared") or next((p for p in CLOUDFLARED_PATHS if os.path.isfile(p)), None)


def cloudflare_link(port, exe, wait=40):
    """Public link through a Cloudflare quick tunnel (https://....trycloudflare.com, no account needed).
    The tunnel runs while the zoo runs. None if no link came within `wait` seconds."""
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:%d" % port],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    atexit.register(proc.terminate)
    found = queue.Queue()

    def read_log():                              # keep reading, or cloudflared stalls once the pipe is full
        for line in proc.stderr:
            m = re.search(r"https://(?!api\.)[-a-z0-9]+\.trycloudflare\.com", line)
            if m:
                found.put(m.group(0))
        found.put(None)

    threading.Thread(target=read_log, daemon=True).start()
    try:
        url = found.get(timeout=wait)
    except queue.Empty:
        url = None
    if not url:
        proc.terminate()
    return url


def share_failed_help():
    """Why Gradio's public link failed, and what to do instead."""
    try:
        from gradio.tunneling import BINARY_PATH
        frpc_missing = not os.path.exists(BINARY_PATH)
    except Exception:
        frpc_missing = False
    if frpc_missing:
        why = ("Gradio's link program (frpc) is gone: an antivirus (e.g. Windows Defender, common on work PCs)\n"
               "  most likely deleted it right after the download, or the download failed (no internet).\n"
               "  Each new try downloads it again and gets flagged again.")
    else:
        why = ("The Gradio link server could not be reached (a firewall / proxy blocks it, or it is down:\n"
               "  https://status.gradio.app).")
    return ("Public link: could not be created. %s\n"
            "  Ways around it:\n"
            "  - install Cloudflare's free tunnel program once, then the zoo uses that instead (see README):\n"
            "      Windows: winget install --id Cloudflare.cloudflared    macOS: brew install cloudflared\n"
            "  - or LAN = True at the top of main.py (or --lan): a link for phones / PCs on the same network" % why)


def launch(model, device, threshold, iou, output_dir, example_image, port=None, prompt=None, classifier="auto",
           classes=None, share=False, lan=False):
    import gradio as gr

    demo = build(model, device, threshold, iou, output_dir, example_image, prompt, classifier, classes)
    os.makedirs(output_dir, exist_ok=True)
    cloudflared = find_cloudflared() if share else None
    if share:
        print("\nCreating a public link with %s (needs internet, takes a few seconds) ..."
              % ("Cloudflare" if cloudflared else "Gradio"))
    _, local_url, share_url = demo.queue().launch(
        inbrowser=True, server_name="0.0.0.0" if lan else "127.0.0.1", server_port=port,
        allowed_paths=[output_dir], share=share and not cloudflared, quiet=True, prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="green"), footer_links=[], js=page_js(), css=CSS)
    if cloudflared:
        share_url = cloudflare_link(demo.server_port, cloudflared)
    print("\nThe UI is open in your browser: %s" % local_url)
    if lan:
        ip = lan_address()
        if ip:
            lan_url = "http://%s:%d" % (ip, demo.server_port)
            print("Same-network link: %s" % lan_url)
            print("  Phones / PCs on the same network (Wi-Fi) can open it. If Windows asks, allow Python on private\n"
                  "  networks. Guest / campus Wi-Fi (e.g. eduroam) often blocks this; a phone hotspot usually works.")
            print_qr(lan_url)
        else:
            print("Same-network link: this computer is not on a network.")
    if share_url:
        print("Public link: %s" % share_url)
        print("  Anyone with this link can use the zoo on this computer (%s)."
              % ("until you stop it" if cloudflared else "it lasts up to a week, or until you stop"))
        print_qr(share_url)
    elif cloudflared:
        print("Public link: Cloudflare gave no link (no internet, or blocked by a firewall / proxy).\n"
              "  LAN = True at the top of main.py (or --lan) gives a link for phones / PCs on the same network.")
    elif share:
        print(share_failed_help())
    else:
        print("Public link: off (to share the UI with others: SHARE = True at the top of main.py, or --share)")
    print("Results are also saved to %s. Press Ctrl+C here to stop.\n" % output_dir)
    demo.block_thread()
