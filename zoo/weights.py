"""
Weight download + cache.

Weights are NOT stored in this repository. The first time a model is used, its weight file is downloaded from the
original authors' source (the URL in zoo/registry.py) into weights/<model-name>/, the SHA-256 checksum is verified,
and every later run uses that local copy (works offline). Delete the folder to force a new download.

Gated models (e.g. SAM3) are downloaded from Hugging Face with your token (HF_TOKEN in main.py, the HF_TOKEN
environment variable, or a saved `hf auth login`). See docs/GATED_MODELS.md.
"""

import hashlib
import os
import shutil
import time

from .registry import GATED_GUIDE_URL, MB, size_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.environ.get("INSECT_ZOO_WEIGHTS", os.path.join(HERE, "weights"))


class GatedModelError(RuntimeError):
    """Access to a gated model is missing; the message says exactly what to do."""


def hf_token():
    """HF_TOKEN from main.py / the environment, else the token saved by `hf auth login` (if any)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return (get_token() or "").strip()
    except Exception:
        return ""


def gated_help(card, problem):
    return "\n".join([
        problem,
        "",
        "'%s' is a GATED model: it is free, but its authors must approve your access first," % card.name,
        "and the zoo needs your Hugging Face token to download it.",
        "  1. Request access (log in first):  %s" % card.gated,
        "  2. Create a fine-grained token:     https://huggingface.co/settings/tokens",
        "     tick 'Read access to contents of all public gated repos you can access'",
        '  3. Paste it at the top of main.py:  HF_TOKEN = "hf_..."',
        "  4. Run the same command again.",
        "Step-by-step guide: %s" % GATED_GUIDE_URL])


def weight_path(card):
    return os.path.join(WEIGHTS_DIR, card.name, card.weights.filename)


def is_downloaded(card):
    p = weight_path(card)
    return os.path.isfile(p) and os.path.getsize(p) == card.weights.size


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tqdm_progress(desc):
    """Default progress reporter for the terminal."""
    from tqdm import tqdm
    bar = {}

    def report(done, total, _msg):
        if "bar" not in bar:
            bar["bar"] = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
        bar["bar"].update(done - bar["bar"].n)
        if done >= total:
            bar["bar"].close()
    return report


def ensure_weights(card, progress=None):
    """Return the local path of the model's weight file, downloading it first if needed.

    progress(done_bytes, total_bytes, message) is called while downloading (the UI passes its own; the CLI gets a
    tqdm bar)."""
    dest = weight_path(card)
    if is_downloaded(card):
        return dest

    token = hf_token() if card.gated else ""
    if card.gated and not token:
        raise GatedModelError(gated_help(card, "No Hugging Face token found."))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(dest)).free
    if free < card.weights.size + 200 * MB:
        raise RuntimeError("Not enough free disk space in %s for %s (%s needed, %s free)."
                           % (WEIGHTS_DIR, card.weights.filename, size_text(card.weights.size), size_text(free)))

    progress = progress or _tqdm_progress(card.weights.filename)
    tmp = dest + ".part"
    errors = []
    for url in card.weights.urls:
        try:
            _download(url, tmp, card.weights.size, card.weights.filename, progress, token, card)
        except GatedModelError:
            raise
        except Exception as e:              # network error, 404, ... -> try the next mirror
            errors.append("%s: %s" % (url, e))
            continue
        digest = sha256(tmp)
        if digest != card.weights.sha256:
            os.remove(tmp)
            errors.append("%s: checksum mismatch (got %s)" % (url, digest))
            continue
        os.replace(tmp, dest)
        return dest
    raise RuntimeError("Could not download %s for model '%s':\n  %s\nCheck your internet connection, or download it "
                       "manually and place it at %s" % (card.weights.filename, card.name, "\n  ".join(errors), dest))


def _download(url, tmp, expected_size, name, progress, token="", card=None):
    import requests

    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": "bytes=%d-" % done} if 0 < done < expected_size else {}
    if token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = "Bearer " + token     # requests drops it when redirected to the file CDN
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if card is not None and card.gated and r.status_code == 401:
            raise GatedModelError(gated_help(card, "Hugging Face did not accept your token (error 401). Check that "
                                                   "it was copied completely and has not been deleted."))
        if card is not None and card.gated and r.status_code == 403:
            raise GatedModelError(gated_help(card, "Your token works, but it cannot download '%s' yet (error 403). "
                                                   "Either your access request is still pending (check "
                                                   "https://huggingface.co/settings/gated-repos), or the token is "
                                                   "missing the 'public gated repos' permission." % card.name))
        r.raise_for_status()
        if r.status_code != 206:                # server ignored the resume request -> start over
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done or expected_size
        last = 0.0
        with open(tmp, "ab" if done else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.2 or done >= total:
                    progress(done, total, "Downloading %s  %.0f / %.0f MB" % (name, done / MB, total / MB))
                    last = now
