"""
Weight download + cache.

Weights are NOT stored in this repository. The first time a model is used, its weight file is downloaded from the
original authors' source (the URL in zoo/registry.py) into weights/<model-name>/, the SHA-256 checksum is verified,
and every later run uses that local copy (works offline). Delete the folder to force a new download.

Gated models (e.g. SAM3) are downloaded from Hugging Face with your token: the HF_TOKEN environment variable (also
set by HF_TOKEN in main.py), the file hf_token.txt next to main.py (git-ignored), or a saved `hf auth login`.
See docs/GATED_MODELS.md.
"""

import hashlib
import os
import shutil
import socket
import threading
import time
from urllib.parse import urlparse

from .registry import GATED_GUIDE_URL, MB, display_name, size_text

DOWNLOAD_TRIES = 3          # per mirror: tries in a row that get no data (a dropped download resumes where it stopped)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.environ.get("INSECT_ZOO_WEIGHTS", os.path.join(HERE, "weights"))


class GatedModelError(RuntimeError):
    """Access to a gated model is missing; the message says exactly what to do."""


TOKEN_FILE = os.path.join(HERE, "hf_token.txt")          # git-ignored, so a token there is never uploaded


def hf_token():
    """HF_TOKEN from the environment (or main.py), else hf_token.txt, else the token saved by `hf auth login`."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            token = f.read().strip()
        if token:
            return token
    except OSError:
        pass
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
        "  3. Save it: run  hf auth login  and paste it (or put it in hf_token.txt next to main.py)",
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
            if "bar" in bar:                            # end the bar's line first, or the message is glued to it
                bar.pop("bar").close()
            if msg:
                print(msg)
            return
        if "bar" not in bar:                            # a resumed download starts where the last one stopped
            bar["bar"] = tqdm(total=total, initial=done, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
        bar["bar"].update(done - bar["bar"].n)
        if done >= total:
            bar.pop("bar").close()
    return report


def ensure_weights(card, progress=None):
    """Return the local path of the model's weight file, downloading it (and its small extra files) first if needed.

    progress(done_bytes, total_bytes, message) is called while downloading (the UI passes its own; the CLI gets a
    tqdm bar)."""
    _check_folder_and_space(card)
    files = (card.weights,) + tuple(card.species_table)   # BioCLIP: + every taxon it knows (used when you give no names)
    paths = [_ensure_file(card, wf, _numbered(progress, n, len(files))) for n, wf in enumerate(files, 1)]
    _ensure_extra_files(card, progress)
    return paths[0]


def _numbered(progress, n, count):
    """With several files (BioCLIP: weights + species table), every message says which one, e.g. '(file 2 of 3)':
    otherwise the size count starting again from 0 looks like the download started over."""
    if progress is None or count == 1:
        return progress
    return lambda done, total, msg: progress(done, total, "%s (file %d of %d)" % (msg, n, count))


def _check_folder_and_space(card):
    """Before downloading anything: the weights folder can be used, and has room for everything this model still
    needs (weights + species table), not just the next file."""
    missing = sum(wf.size for wf in (card.weights,) + tuple(card.species_table) if not _have(card, wf))
    if not missing:
        return
    folder = os.path.join(WEIGHTS_DIR, card.name)
    try:
        os.makedirs(folder, exist_ok=True)
        free = shutil.disk_usage(folder).free
    except OSError as e:
        raise RuntimeError("The weights folder %s cannot be used (%s). If it is on a drive that is not plugged in, "
                           "plug it in; or choose another folder: WEIGHTS_DIR at the top of main.py, --weights_dir, "
                           "or the INSECT_ZOO_WEIGHTS environment variable." % (WEIGHTS_DIR, e))
    if free < missing + 200 * MB:
        raise RuntimeError("Not enough free disk space for %s: %s needed, %s free in %s. Free up space, or keep the "
                           "weights on another drive: WEIGHTS_DIR at the top of main.py or --weights_dir."
                           % (card.name, size_text(missing), size_text(free), WEIGHTS_DIR))


def weights_status(card):
    """'downloaded' or 'not downloaded (size)', for --check."""
    if is_downloaded(card):
        return "downloaded"
    missing = sum(wf.size for wf in (card.weights,) + tuple(card.species_table) if not _have(card, wf))
    return "not downloaded (%s)" % size_text(missing)


def _ensure_extra_files(card, progress=None):
    """Small helper files some models need next to their weights (e.g. the upstream classifier code), pinned URLs."""
    import requests
    todo = [(rel, url) for rel, url in card.extra_files
            if not os.path.exists(os.path.join(WEIGHTS_DIR, card.name, *rel.split("/")))]
    for n, (rel, url) in enumerate(todo, 1):
        dest = os.path.join(WEIGHTS_DIR, card.name, *rel.split("/"))
        if progress:
            progress(0, 0, "Downloading %s: helper file %d of %d (%s)" % (display_name(card), n, len(todo),
                                                                          rel.split("/")[-1]))
        _avoid_dead_ipv6(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for attempt in range(1, DOWNLOAD_TRIES + 1):
            try:
                r = requests.get(url, timeout=(10, 60))
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == DOWNLOAD_TRIES or not _transient(e):
                    raise RuntimeError("Could not download %s: %s. %s.\nDetails: %s: %s"
                                       % (display_name(card), _why(e), _advice(e), url, e))
                time.sleep(5 * attempt)
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
        if os.path.isfile(tmp) and os.path.getsize(tmp) == wf.size:     # complete, but the app closed before the
            progress(0, 0, "Checking the download of %s" % display_name(card))   # check: no need to download again
            if sha256(tmp) == wf.sha256:
                os.replace(tmp, dest)
                return dest
            os.remove(tmp)
        errors, last = [], None
        for url in wf.urls:
            try:
                _download_with_retries(url, tmp, wf, progress, token, card, lock)
            except GatedModelError:
                raise
            except Exception as e:          # network error, 404, ... -> try the next mirror
                errors.append("%s: %s" % (url, e))
                last = e
                continue
            progress(0, 0, "Checking the download of %s" % display_name(card))
            digest = sha256(tmp)
            if digest != wf.sha256:
                os.remove(tmp)
                errors.append("%s: checksum mismatch (got %s)" % (url, digest))
                last = None
                continue
            os.replace(tmp, dest)
            return dest
    finally:
        _release_lock(lock)
    progress(0, 0, "")                      # the terminal's progress bar ends before the error
    kept = os.path.isfile(tmp) and os.path.getsize(tmp) > 0
    reason = _why(last) if last is not None else "the downloaded file was damaged (checksum mismatch)"
    # the first line is a whole sentence: the UI shows only that line, the terminal gets all of it
    raise RuntimeError("Could not download %s: %s. %s%s.\nDetails:\n  %s\nOr download %s yourself and place it at %s"
                       % (display_name(card), reason, _advice(last),
                          " (it continues where it stopped)" if kept and _transient(last) else "",
                          "\n  ".join(errors), wf.urls[0], dest))


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


def _download_with_retries(url, tmp, wf, progress, token, card, lock):
    """A dropped connection, a stall (no data for 60 s) or a server error is retried after a short pause; the
    .part file is kept, so every try resumes where the last one stopped. Only tries that got no data count towards
    DOWNLOAD_TRIES: a big download on a flaky line keeps going as long as it makes progress (at most 20 tries).
    Other errors (404, a blocked certificate check, ...) fail at once."""
    failed = 0
    for attempt in range(1, 21):
        before = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            return _download(url, tmp, wf.size, wf.filename, progress, token, card, lock)
        except GatedModelError:
            raise
        except Exception as e:
            grew = os.path.exists(tmp) and os.path.getsize(tmp) > before
            failed = 0 if grew else failed + 1
            if failed >= DOWNLOAD_TRIES or attempt == 20 or not _transient(e):
                raise
            progress(0, 0, "Downloading %s: %s, trying again (%s)"
                     % (display_name(card), _why(e), "try %d of %d" % (failed + 1, DOWNLOAD_TRIES) if failed
                        else "it continues where it stopped"))
            if lock:
                os.utime(lock)              # still alive: other windows keep waiting instead of taking over
            time.sleep(5 * max(failed, 1))


_ipv6_checked = set()
_ipv6_lock = threading.Lock()


def _avoid_dead_ipv6(url):
    with _ipv6_lock:                            # the start-up check and a download may ask at the same time
        _check_ipv6(url)


def _check_ipv6(url):
    """Some networks give out IPv6 addresses that lead nowhere. Python then waits for a timeout on every IPv6 address
    of the server (GitHub has four: minutes) before it tries IPv4. So once per server: if its IPv6 does not answer
    within 3 s but its IPv4 does, use IPv4 only for the rest of this run."""
    host = urlparse(url).hostname
    if not host or host in _ipv6_checked:
        return
    from urllib3.util import connection
    if not connection.HAS_IPV6:
        return

    def answers(family, timeout):
        try:
            ip = socket.getaddrinfo(host, 443, family, socket.SOCK_STREAM)[0][4][0]
            socket.create_connection((ip, 443), timeout=timeout).close()
            return True
        except OSError:
            return False

    # A host only counts as checked once there is an answer: started offline (or before Wi-Fi is up), the next
    # download checks again instead of waiting on dead IPv6 addresses for the rest of the run
    try:
        infos = socket.getaddrinfo(host, 443, 0, socket.SOCK_STREAM)
    except OSError:
        return                                  # offline / DNS failed: not marked, checked again next time
    if not any(a[0] == socket.AF_INET6 for a in infos):
        _ipv6_checked.add(host)                 # no IPv6 address for this server: nothing to avoid
    elif answers(socket.AF_INET6, 3):
        _ipv6_checked.add(host)                 # IPv6 works here
    elif answers(socket.AF_INET, 5):
        _ipv6_checked.add(host)
        print("NOTE: IPv6 does not work on this network (%s did not answer), downloading over IPv4." % host)
        connection.HAS_IPV6 = False             # urllib3 (requests) then only looks up IPv4 addresses
        _ipv4_first()                           # ... and every other library (Hugging Face, torch hub, ...)
    # neither answered (offline, captive portal, no route yet): not marked, so the next download checks again


def _ipv4_first():
    """Process-wide: put IPv4 addresses first, so no library waits on dead IPv6 ones (IPv6-only hosts still work)."""
    if getattr(socket.getaddrinfo, "ipv4_first", False):
        return
    original = socket.getaddrinfo

    def getaddrinfo(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda a: a[0] != socket.AF_INET)
    getaddrinfo.ipv4_first = True
    socket.getaddrinfo = getaddrinfo


def check_network_in_background():
    """At start: test IPv6 to Hugging Face and GitHub (where the weights come from) without making anyone wait."""
    def check():
        for url in ("https://huggingface.co/", "https://raw.githubusercontent.com/"):
            try:
                _avoid_dead_ipv6(url)
            except Exception:
                pass
    threading.Thread(target=check, name="network-check", daemon=True).start()


def _blocked(e):
    """The network itself is in the way: a proxy refused, or the certificate check failed (a company or university
    network, or an antivirus, inspecting HTTPS). Trying again does not help."""
    import requests
    return isinstance(e, requests.exceptions.ProxyError) or "certificate verify failed" in str(e).lower()


def _why(e):
    """A dropped / stalled / refused download in a few plain words."""
    import requests
    if isinstance(e, requests.exceptions.ProxyError):
        return "the proxy refused the connection"
    if _blocked(e):
        return ("the secure connection was blocked (the certificate check failed: a company or university network, "
                "or an antivirus, may be inspecting HTTPS)")
    if isinstance(e, requests.HTTPError) and e.response is not None:
        code = e.response.status_code
        return ("the file is not on the server (error 404)" if code == 404 else
                "the server had a problem (error %d)" % code if code >= 500 else
                "the server refused the download (error %d)" % code)
    if isinstance(e, (requests.Timeout, TimeoutError)):
        return "the server stopped answering"
    if isinstance(e, (requests.RequestException, ConnectionError)):
        return "the connection dropped"
    if isinstance(e, OSError):                  # writing the file: disk full, no permission, ...
        return "the file could not be saved (%s)" % (e.strerror or e)
    return "the download failed (%s)" % type(e).__name__


def _advice(e):
    """What to do about it, for the error message."""
    import requests
    if e is not None and _blocked(e):
        return "Ask your IT support, or download the file yourself (see below)"
    if isinstance(e, OSError) and not isinstance(e, (requests.RequestException, ConnectionError, TimeoutError)):
        return "Check the free space and the weights folder, then try again"
    return "Check your internet connection and try again"


def _transient(e):
    """Worth another try: the connection dropped or stalled, or the server had a (temporary) problem."""
    import requests
    if e is None or _blocked(e):
        return False
    if isinstance(e, requests.HTTPError):
        code = e.response.status_code if e.response is not None else 0
        return code == 429 or code >= 500
    return isinstance(e, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                          ConnectionError, TimeoutError))


def _download(url, tmp, expected_size, name, progress, token="", card=None, lock=None):
    import requests

    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": "bytes=%d-" % done} if 0 < done < expected_size else {}
    if token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = "Bearer " + token     # requests drops it when redirected to the file CDN
    progress(0, 0, "%s %s: connecting to %s" % ("Resuming" if done else "Downloading", display_name(card) if card
                                                 else name, urlparse(url).hostname))
    _avoid_dead_ipv6(url)
    with requests.get(url, stream=True, timeout=(10, 60), headers=headers) as r:
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
