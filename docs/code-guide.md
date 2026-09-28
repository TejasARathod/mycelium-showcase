# Code guide

[← Overview](../README.md)

| Component | Included file | Status |
|---|---|---|
| Distributional image evaluation | `src/mycelium/distributional_metrics.py` | Reusable CLI; requires ML dependencies and model downloads |
| Controlled experiment prompts | `src/mycelium/prompts.py` | Importable without ML dependencies; final `N_SEEDS=250` |
| LoRA training wrapper | `reference/training/train_lora.py` | Historical GPU reference; expects a Diffusers training script under `/workspace` |
| Generator attention diagnostics | `reference/explainability/explain_checkpoint.py` | Historical CUDA reference; includes checkpoint loading, attention capture and scoring |
| GPU evaluation orchestration | `reference/orchestration/orchestrate.py` | Historical reference; requires original config, companion scripts, SSH and RunPod |
| Figure regeneration | `scripts/plot_results.py` | Local CPU; reads included tables |

## Why selected source is retained unchanged

The original research scripts contain experimental context and bug-fix history. They have been copied byte-for-byte, with SHA-256 provenance, rather than silently rewritten into a different experiment. Comments may mention old sample budgets or archive paths: the final prompt constant is 250, and the submitted thesis is authoritative for final results.

The orchestration reference demonstrates the push/run/pull sequence, model-specific environments and teardown logic. It is Python orchestration; no LangGraph implementation has been verified in the selected source. It is not a standalone launch command in this release.

`distributional_metrics.py` labels its polynomial-kernel MMD output `kid`; because it uses normalised CLIP/DINOv2 embeddings, interpret it as that implementation's feature-space MMD rather than standard Inception-feature KID. This release preserves the historical naming and does not make it a headline metric.

## Original archive locations

The [manifest](code-provenance.json) maps every selected file to its original path and records its hash. The original E06 dataset, training, generation and evaluation folders contain the full experiment context. Those scripts and their datasets remain in the research archive; the showcase does not imply a turnkey reproduction of all results.
