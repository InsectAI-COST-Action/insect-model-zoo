# How weights are downloaded

[← back to the README](../README.md)

Weights are **not stored in this repository**. The files stay with the original authors, and the zoo stays small.

1. The first time a model is used (UI, command line or `--download`), `main.py` downloads its weight file from the
   original source listed in [`zoo/registry.py`](../zoo/registry.py). You see a progress bar in the terminal or the UI.
2. The file is saved to `weights/<model>/`. An interrupted download resumes where it stopped.
3. The **SHA-256 checksum** is checked, so you get exactly the file this zoo was tested with. If the upstream file
   ever changes, you get a clear error instead of silently different results.
4. Every later run uses the local copy, so it **works offline**. To get everything in advance (fieldwork, cluster
   login node, slow connection), run `python main.py --download all`: detectors ≈ 270 MB, classifiers ≈ 11.5 GB
   (insectDCT 484 MB, BioCLIP 2.5 6.8 GB and BioCLIP 2 4.2 GB, each with its species table), plus 3.2 GB for `sam3`
   once you have access (skipped without). Or download just one: `python main.py --download bioclip-2`.
5. If two windows (e.g. the UI and a command line) need the same model at once, the second one waits for the first
   download to finish instead of downloading it twice.
6. **Gated models** (`sam3`) download from Hugging Face with your token: see [docs/GATED_MODELS.md](GATED_MODELS.md).

Delete `weights/<model>/` to download a model again. To keep weights somewhere else (e.g. an external or shared
drive): `WEIGHTS_DIR` at the top of `main.py`, `--weights_dir`, or the environment variable
`INSECT_ZOO_WEIGHTS=/path/to/folder` (see [Weights on another drive](../README.md#weights-on-another-drive)). Before a
download starts, the zoo checks that the folder can be used and has room for the whole model (weights and, for BioCLIP,
its species table), and `python main.py --check` shows the folder, its free space and which models are downloaded. Each model can have several URLs that are tried in order,
so a mirror can be added later without changing anything else.
