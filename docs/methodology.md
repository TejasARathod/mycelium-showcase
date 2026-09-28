# Methodology

[← Overview](../README.md)

## Research design

The final comparison uses **FLUX.2 klein Base 9B** across India and the UK, with rural roads and urban commercial streets. For each cell, compare held-out real photographs, the base generator and the same generator with a cell-specific LoRA adapter.

| Component | Final experiment |
|---|---|
| Training set | 40 images per adapter: 20 real OSV-5M photographs and 20 AI-generated images derived from additional real photographs |
| Training | 800 steps at 320 × 320; checkpoints every 100 steps |
| Adapter | Rank 16; learning rate 1e-4 in the recorded training wrapper |
| Checkpoint selection | Blind model-judge image comparisons; not a human study |
| Generation | 250 seeds per cell/condition; five prompt paraphrases rotated across seeds |
| Comparison control | Identical prompts, seeds, steps and guidance with/without LoRA; no added trigger token |
| Output settings | 1024 × 1024, 50 steps, guidance 8.0 |
| Planned generated total | 4 cells × 2 conditions × 250 = 2,000 images |
| Real reference | 250 images per cell; geographically separated from real training images by at least 1 km, with near-duplicate checks |
| Uncertainty | 5,000 bootstrap resamples for reported per-image measures |

## Evaluation layers

1. **Geographic identity:** GeoReasoner country accuracy and PLONK GeoScore.
2. **Local detail:** object-inventory comparisons and model-judge locale-specificity ratings.
3. **Distributional similarity:** precision, recall and coverage in CLIP and DINOv2 representations.
4. **Mechanistic diagnostics:** token/attention interventions and evaluator-specific tests.

The metrics answer different questions. Their agreement is interpreted in light of representation dependencies. We retain component results rather than making an overall composite the headline claim.

## Development process

The research included controlled investigations of caption strategy, prior preservation, adapter rank, learning rate and dataset composition. A pre-declared CLIP-I gate was used for a particular trigger-word study; it is an experimental decision rule, not a generally calibrated measure of geographic fidelity. Final results are reported separately from those exploratory tests.

## Scope of conclusions

The study covers two countries, two scene types and one final generator family. Confidence intervals quantify sampling variability within the experiment, not geographic generalisation. Image corpora, adapter weights and full GPU environment locks are not distributed in this curated release.
