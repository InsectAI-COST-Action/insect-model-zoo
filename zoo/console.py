"""Terminal output that can never pause the web UI.

On Windows, clicking or selecting text in the terminal window ("QuickEdit") pauses every program that prints to it
until you press Esc or Enter (Ctrl+S does the same on macOS / Linux). A model loading in the background would then
stop at its next print and the UI would wait forever. With nonblocking() every print goes into a queue and a
separate thread writes it to the terminal: only that thread waits while the terminal is paused, the models keep
loading, and the paused lines all appear once the terminal is released.
"""

import atexit
import queue
import sys
import threading
import time

_queue = queue.Queue()


class _Writer:
    """Stands in for sys.stdout / sys.stderr: write() only queues the text."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if text:
            _queue.put((self._stream, text))
        return len(text)

    def flush(self):
        pass

    def __getattr__(self, name):            # encoding, isatty, fileno, ... come from the real stream
        return getattr(self._stream, name)


def _pump():
    while True:
        stream, text = _queue.get()
        try:
            stream.write(text)
            stream.flush()
        except Exception:
            pass


def _drain(seconds=2.0):
    """At exit: give queued lines a moment to reach the terminal, but never hang on a paused one."""
    end = time.time() + seconds
    while not _queue.empty() and time.time() < end:
        time.sleep(0.05)


def nonblocking():
    """Route sys.stdout and sys.stderr through the queue (one queue, so their lines stay in order)."""
    if isinstance(sys.stdout, _Writer):
        return
    threading.Thread(target=_pump, name="console", daemon=True).start()
    sys.stdout, sys.stderr = _Writer(sys.stdout), _Writer(sys.stderr)
    atexit.register(_drain)
