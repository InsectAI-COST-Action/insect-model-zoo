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

from .registry import GATED_GUIDE_URL, MB, display_name, size_text

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


def weight_path(card, wf=None):
    """Local path of a model's weight file (or of another of its files, e.g. a species table)."""
    return os.path.join(WEIGHTS_DIR, card.name, (wf or card.weights).filename)


def _have(card, wf):
    p = weight_path(card, wf)
    return os.path.isfile(p) and os.path.getsize(p) == wf.size


def is_downloaded(card):
    """Weights and (zero-shot classifiers) species table are all there."""
    return all(_have(card, wf) for wf in (card.weights,) + tuple(card.species_table))


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

    def report(done, total, msg):
        if not total:                                   # a status message, not a byte count
            print(msg)
            return
        if "bar" not in bar:
            bar["bar"] = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
        bar["bar"].update(done - bar["bar"].n)
        if done >= total:
            bar["bar"].close()
    return report


def ensure_weights(card, progress=None):
    """Return the local path of the model's weight file, downloading it (and its small extra files) first if needed.

    progress(done_bytes, total_bytes, message) is called while downloading (the UI passes its own; the CLI gets a
    tqdm bar)."""
    path = _ensure_file(card, card.weights, progress)
    for wf in card.species_table:                   # BioCLIP: every taxon it knows (used when you give no names)
        _ensure_file(card, wf, progress)
    _ensure_extra_files(card)
    return path


def _ensure_extra_files(card):
    """Small helper files some models need next to their weights (e.g. the upstream classifier code), pinned URLs."""
    import requests
    for rel, url in card.extra_files:
        dest = os.path.join(WEIGHTS_DIR, card.name, *rel.split("/"))
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dest + ".part", dest)


def _ensure_file(card, wf, progress):
    dest = weight_path(card, wf)
    if _have(card, wf):
        return dest

    token = hf_token() if card.gated else ""
    if card.gated and not token:
        raise GatedModelError(gated_help(card, "No Hugging Face token found."))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(dest)).free
    if free < wf.size + 200 * MB:
        raise RuntimeError("Not enough free disk space in %s for %s (%s needed, %s free)."
                           % (WEIGHTS_DIR, wf.filename, size_text(wf.size), size_text(free)))

    progress = progress or _tqdm_progress(wf.filename)
    lock = dest + ".lock"
    _acquire_lock(lock, progress, display_name(card))
    try:
        if _have(card, wf):                 # another window finished it while we waited
            return dest
        tmp = dest + ".part"
        errors = []
        for url in wf.urls:
            try:
                _download(url, tmp, wf.size, wf.filename, progress, token, card, lock)
            except GatedModelError:
                raise
            except Exception as e:          # network error, 404, ... -> try the next mirror
                errors.append("%s: %s" % (url, e))
                continue
            digest = sha256(tmp)
            if digest != wf.sha256:
                os.remove(tmp)
                errors.append("%s: checksum mismatch (got %s)" % (url, digest))
                continue
            os.replace(tmp, dest)
            return dest
    finally:
        _release_lock(lock)
    raise RuntimeError("Could not download %s for model '%s':\n  %s\nCheck your internet connection, or download it "
                       "manually and place it at %s" % (wf.filename, card.name, "\n  ".join(errors), dest))


def _acquire_lock(lock, progress, name):
    """One download per file at a time, also across windows/processes (UI + command line). Others wait for it.
    The lock holds the owner's process id, so a lock left by a closed or crashed window is taken over at once."""
    import psutil
    told = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            pass
        try:
            with open(lock) as f:
                owner = int(f.read().strip() or 0)
            stale = not psutil.pid_exists(owner) or time.time() - os.path.getmtime(lock) > 300
        except (OSError, ValueError):
            continue                        # the lock vanished or is being written: just try again
        if stale:
            _release_lock(lock)
            continue
        if not told:
            progress(0, 0, "Waiting: %s is already being downloaded in another window ..." % name)
            told = True
        time.sleep(1)


def _release_lock(lock):
    try:
        os.remove(lock)
    except OSError:
        pass


def _download(url, tmp, expected_size, name, progress, token="", card=None, lock=None):
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
                    if lock:
                        os.utime(lock)          # "still alive" for other windows waiting on this download
