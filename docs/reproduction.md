# Reproduction guide

[← Overview](../README.md)

## Level 1 — review and regenerate the published summaries

Requires Python 3.10+ and Matplotlib. No GPU, credentials or model downloads.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/plot_results.py
python -m unittest discover -s tests -v
```

This rebuilds the result plots from transcribed thesis tables. It does not recompute image-level model scores or bootstrap intervals.

## Level 2 — evaluate your own image directories

Install PyTorch/torchvision appropriate to your hardware and the remaining packages in `requirements-analysis.txt`. These are dependency names, not a locked reproduction of the original GPU environment.

```bash
PYTHONPATH=src python -m mycelium.distributional_metrics   --real-dir /path/to/real --gen-dir /path/to/generated   --backbone clip --device cpu --k 3 --out-dir outputs/clip
```

Repeat with `--backbone dinov2` for the second representation. First use downloads external model weights; DINOv2 uses `torch.hub`. Use readable JPG/PNG files and more than `k` images in each directory. Scores depend on the selected images, preprocessing and model versions. This CLI was syntax-checked but was not run with model weights during release preparation.

## Level 3 — original training and GPU orchestration

The `reference/` scripts document the implementation, but need their original companion code, dataset manifests, configuration, model permissions and compatible GPU environment. Do not execute them as a quickstart. The training wrapper has a hard-coded Diffusers example path; the orchestrator expects the archive's config and remote scripts. Its historical GPU rate estimates are not current pricing.

Full end-to-end reproduction is not claimed by this release. No GPU resources were launched during its preparation. Credentials should be supplied externally, never written into Git.

## What is included and what remains external

Included: selected source, controlled prompts, aggregate tables, derived plots and methodology. External: OSV-5M source imagery, model weights, four LoRA adapters, raw per-image evaluation outputs and full environment locks. Consult upstream usage conditions before downloading or redistributing data and models.

The orchestration reference requires JUPYTER_PASSWORD from the environment; its historical hard-coded demo value has been removed for publication.
