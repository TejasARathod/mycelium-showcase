# Results and provenance

These small tables are transcribed from the **submitted MSc thesis**, not recomputed from raw images in this release.

| File | Source | Interpretation |
|---|---|---|
| `geographic_scores.csv` | Table 13, printed page 44 | GeoReasoner correct/valid counts and mean PLONK GeoScore |
| `india_rural_intervals.json` | Table 14, printed page 45 | Means and 95% bootstrap intervals; 5,000 resamples |
| `distributional_metrics.csv` | Table 10, printed page 41 | CLIP/DINOv2 precision, recall and coverage |

Two base image sets have 249 valid images after one truncated PNG was excluded from each. Keep the explicit denominators rather than reporting every group as 250. The bootstrap intervals describe within-condition resampling; they are not a multiple-comparison-adjusted hypothesis test.

PLONK GeoScore, geolocation accuracy and distributional recall measure different properties. A PLONK score is not a percentage accuracy. Locale-specificity is a model-judge rating, not a human study.

Figure scripts read these files directly. The full image corpus, adapters and raw per-image scores are not bundled. No new training or inference was run to prepare this release.
