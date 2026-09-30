# Troubleshooting

[← back to the README](../README.md)

| Problem | Fix |
|---|---|
| `python` is not recognized / opens the Microsoft Store | Python is not on the PATH; see [Prerequisites](../README.md#prerequisites) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | see the tip under [Installation](../README.md#installation) |
| `error: externally-managed-environment` | the virtual environment is not active: activate `.venv` first |
| `DLL load failed ... The filename or extension is too long` (Windows) | the folder path is too long: move the project to e.g. `C:\code\insect-model-zoo`, delete `.venv` and create it again |
| `ImportError: libGL.so.1` (Linux server / WSL) | `sudo apt install libgl1 libglib2.0-0` |
| Runs on CPU although you have an NVIDIA GPU | `python main.py --check` tells you why; usually the CPU-only PyTorch, see [GPU](HARDWARE.md#getting-the-gpu-to-work) |
| *The weights folder ... cannot be used* | the drive with the weights is not plugged in, or the folder is wrong: plug it in, or set `WEIGHTS_DIR` in `main.py` / `--weights_dir` (see [Weights on another drive](../README.md#weights-on-another-drive)) |
| *Not enough free disk space* | the zoo checks the whole model before downloading: free up space, or keep the weights on another drive (`WEIGHTS_DIR`) |
| `No module named 'transformers'` (Grounding DINO) | install the requirements again: `python -m pip install -r requirements.txt` |
| Download fails | check the internet connection or proxy; the error shows the URL and where to put a manually downloaded file |
| Too slow | use a GPU, or smaller models (`insectdct-v8-s`, `flatbug-n`, `flatbug-s`, `bioclip-2`), or `-c none` |
| Too many / too few detections | raise / lower `--threshold` (or the *Detection confidence* slider in the UI) |
| BioCLIP gives odd names | give it a list that matches what can be in your images (`--classes`, or the names field in the UI) |
| BioCLIP calls a flower or leaf an insect | by default it only picks from insects: set `BIOCLIP_INSECT_TAXA = False` in `main.py` (see [BioCLIP](MODELS.md#bioclip-25-and-bioclip-2-zero-shot)) |
| `sam3`: *No Hugging Face token found* / *error 401* / *error 403* | see [docs/GATED_MODELS.md](GATED_MODELS.md#if-it-does-not-work) |
| UI on a remote Linux server / cluster (no screen) | run `python main.py --ui --port 7860` there and `ssh -L 7860:localhost:7860 you@server` from your laptop, then open http://127.0.0.1:7860; or just use the command line |
