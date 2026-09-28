"""
Pod-side: per-checkpoint image generation + the cheap explainability subset (B1 block
sweep + B2 key-vector probe) for the klein-Base-9B flat-loss debugging pass.

B1/B2 logic (AttentionCapture, detect_score, bbox_to_patch_mask, BLOCK_SWEEP_PROMPTS,
KEY_PROBE_PROMPTS) is ported verbatim from klein_base_9b_explainability.ipynb's Section B
cells, already fixed and verified twice (2026-07-23 A6000, 2026-07-29 A100 -- see
verification_log.md) -- not reimplemented, per CLAUDE.md R4 (one way to do a thing).
Chosen as the "cheap subset" (not the full A1-A7+B3-B6 battery) because they're the two
tests that directly probe whether the model's attention/grounding mechanism itself
changes across checkpoints, which is what the flat-loss mystery is actually about.

Also regenerates the same 20-fixed-seed India-urban comparison image set the 07-24
flat-loss run used (RESULTS.md: seeds 0-19, guidance_scale=8.0, 50 steps, same prompt),
so there's a fixed reference per checkpoint for the OpenRouter vision-judge to describe.

    # Pre-training baseline (no LoRA):
    python scripts/generate/explain_checkpoint.py --step 0 \
        --out-dir /workspace/outputs/checkpoint_battery/checkpoint-0

    # A trained checkpoint:
    python scripts/generate/explain_checkpoint.py --step 500 \
        --checkpoint-dir /workspace/persistent/outputs/india_urban_lora/checkpoint-500 \
        --out-dir /workspace/outputs/checkpoint_battery/checkpoint-500
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Verbatim from klein_base_9b_finetune_india_urban.ipynb cell 7 (07-24 flat-loss run's
# own before/after prompt) -- keep identical so images stay comparable across sessions.
COMPARISON_PROMPT = (
    "An urban commercial street in India: a paved street with visible lane markings, "
    "lined on both sides by multi-story buildings with shopfronts and densely packed "
    "signboards and hoardings (illegible), auto-rickshaws, motorcycles and scooters "
    "moving among cars and the occasional bus, tangled overhead wires strung between "
    "utility poles, roadside trees including the occasional palm tree, uneven concrete "
    "sidewalks with a few pedestrians, harsh midday sun under a pale hazy sky, no "
    "legible text or identifiable landmarks, photographed from street level"
)
COMPARISON_GUIDANCE = 8.0
COMPARISON_SEEDS = range(20)  # matches RESULTS.md's seeds 0-19

# Verbatim from klein_base_9b_explainability.ipynb cell 26 (B1)
BLOCK_SWEEP_PROMPTS = [
    ("an auto-rickshaw and a cow on a street in Mumbai, India", "rickshaw"),
    ("a bus and a bicycle on a street in Delhi, India", "bus"),
    ("a red postbox and a black taxi on a street in London, UK", "postbox"),
    ("a market stall and a dog on a street in Jaipur, India", "dog"),
]

# Verbatim from cell 28 (B2) -- block-sweep prompts plus 2 more
KEY_PROBE_PROMPTS = BLOCK_SWEEP_PROMPTS + [
    ("a fruit cart and a rickshaw on a street in Kolkata, India", "cart"),
    ("a food truck and a scooter on a street in Bengaluru, India", "truck"),
]


def load_pipeline(checkpoint_dir: str | None):
    from diffusers import Flux2KleinPipeline

    print(f"Loading {MODEL_REPO} (full pipeline)...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_REPO, torch_dtype=torch.bfloat16, token=HF_TOKEN).to("cuda")
    if checkpoint_dir:
        print(f"Applying LoRA weights from {checkpoint_dir}...")
        pipe.load_lora_weights(checkpoint_dir)
    return pipe


def load_scoring_models():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPModel, CLIPProcessor

    print("Loading Grounding DINO + CLIP (for scoring)...")
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to("cuda").eval()
    return gd_processor, gd_model


@torch.no_grad()
def detect_score(gd_processor, gd_model, image, word, box_threshold=0.2, text_threshold=0.15):
    inputs = gd_processor(images=image, text=[word], return_tensors="pt").to("cuda")
    outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]])[0]
    return (results["scores"].max().item() if len(results["scores"]) else 0.0), results


def bbox_to_patch_mask(bbox, img_size, grid_side):
    """Maps a pixel-space GroundingDINO bbox [x0,y0,x1,y1] onto the square image-token grid."""
    x0, y0, x1, y1 = bbox
    W, H = img_size
    gx0, gx1 = int(x0 / W * grid_side), max(int(x1 / W * grid_side) + 1, int(x0 / W * grid_side) + 1)
    gy0, gy1 = int(y0 / H * grid_side), max(int(y1 / H * grid_side) + 1, int(y0 / H * grid_side) + 1)
    mask = torch.zeros(grid_side, grid_side, dtype=torch.bool)
    mask[gy0:gy1, gx0:gx1] = True
    return mask.flatten()


class AttentionCapture:
    """Captures attention + key tensors for the LAST CONDITIONAL pass per block.

    Verbatim from klein_base_9b_explainability.ipynb (fixed 2026-07-23: OOM from
    recomputing the full attention matrix on every SDPA call; wrong CFG branch captured).
    See verification_log.md for the debugging story. Do not modify without re-verifying
    against that log.
    """

    def __init__(self, transformer, num_inference_steps=50, cfg_passes=2, restrict_query_to=None):
        self.transformer = transformer
        self.captured = {}
        self.current_block = None
        self._hooks = []
        self._orig_sdpa = None
        self._call_count = {}
        self._total_calls = num_inference_steps * cfg_passes
        self._target_call = self._total_calls - 1  # last ODD call = last conditional pass
        self._restrict = restrict_query_to

    def _pre_hook(self, kind, idx):
        def hook(module, inputs):
            self.current_block = (kind, idx)
        return hook

    def __enter__(self):
        for i, block in enumerate(self.transformer.transformer_blocks):
            self._hooks.append(block.register_forward_pre_hook(self._pre_hook("double", i)))
        for i, block in enumerate(self.transformer.single_transformer_blocks):
            self._hooks.append(block.register_forward_pre_hook(self._pre_hook("single", i)))

        self._orig_sdpa = F.scaled_dot_product_attention
        capture_ref = self

        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            out = capture_ref._orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                         is_causal=is_causal, scale=scale, **kw)
            key_id = capture_ref.current_block
            if key_id is not None:
                capture_ref._call_count[key_id] = capture_ref._call_count.get(key_id, 0) + 1
                if capture_ref._call_count[key_id] == capture_ref._target_call:
                    with torch.no_grad():
                        q = query if capture_ref._restrict is None else query[:, :, :capture_ref._restrict, :]
                        d = q.shape[-1]
                        s = scale if scale is not None else 1.0 / (d ** 0.5)
                        attn_w = torch.softmax((q.float() @ key.float().transpose(-2, -1)) * s, dim=-1)
                        capture_ref.captured[key_id] = {
                            "attn": attn_w.mean(dim=1)[0].cpu(),
                            "key": key.float().mean(dim=1)[0].cpu(),
                        }
            return out
        F.scaled_dot_product_attention = patched_sdpa
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._orig_sdpa
        for h in self._hooks:
            h.remove()
        self._hooks = []


def generate_comparison_images(pipe, out_dir: Path):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for seed in COMPARISON_SEEDS:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(prompt=COMPARISON_PROMPT, height=1024, width=1024, num_inference_steps=50,
                     guidance_scale=COMPARISON_GUIDANCE, generator=generator).images[0]
        image.save(images_dir / f"seed{seed:02d}.png")
        print(f"  saved images/seed{seed:02d}.png")


def run_block_sweep(pipe, gd_processor, gd_model, resolution: int | None = None) -> dict:
    N_DOUBLE = len(pipe.transformer.transformer_blocks)
    N_SINGLE = len(pipe.transformer.single_transformer_blocks)
    print(f"{N_DOUBLE} double-stream + {N_SINGLE} single-stream blocks")

    rows = []
    for prompt, word in BLOCK_SWEEP_PROMPTS:
        print(f"  B1: {prompt} (target: {word})")
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1)
        text_seq_len = prompt_embeds.shape[1]

        size_kwargs = {"height": resolution, "width": resolution} if resolution else {}
        with AttentionCapture(pipe.transformer, restrict_query_to=text_seq_len) as cap:
            image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=50, guidance_scale=4.0,
                         generator=torch.Generator("cuda").manual_seed(0), **size_kwargs).images[0]

        gd_score, gd_results = detect_score(gd_processor, gd_model, image, word)
        if gd_score == 0.0 or len(gd_results["boxes"]) == 0:
            print(f"    Grounding-DINO didn't detect '{word}' -- skipping")
            continue
        bbox = gd_results["boxes"][gd_results["scores"].argmax()].tolist()

        messages = [{"role": "user", "content": prompt}]
        text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        enc = pipe.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                              max_length=512, return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        start = text.lower().find(word.lower())
        word_span = [i for i, (s, e) in enumerate(offsets) if start != -1 and e > start and s < start + len(word)]
        if not word_span:
            print(f"    couldn't locate '{word}' in tokenized prompt -- skipping")
            continue

        for kind, n_blocks in (("double", N_DOUBLE), ("single", N_SINGLE)):
            for idx in range(n_blocks):
                key = (kind, idx)
                if key not in cap.captured:
                    continue
                attn = cap.captured[key]["attn"]
                total_seq = attn.shape[1]
                image_seq_len = total_seq - text_seq_len
                if image_seq_len <= 0:
                    continue
                grid_side = int(round(image_seq_len ** 0.5))
                if grid_side * grid_side != image_seq_len:
                    continue
                target_mask = bbox_to_patch_mask(bbox, image.size, grid_side)
                t2i = attn[word_span, text_seq_len:]
                attn_to_target = t2i[:, target_mask].mean().item() if target_mask.any() else float("nan")
                attn_to_all = t2i.mean().item()
                enrichment = attn_to_target / attn_to_all if attn_to_all > 0 else float("nan")
                rows.append({"prompt": prompt, "word": word, "block_kind": kind, "block_idx": idx,
                             "gd_score": gd_score, "enrichment": enrichment})
        del cap
        gc.collect()
        torch.cuda.empty_cache()

    if not rows:
        return {"rows": [], "peak_block": None, "trough_block": None,
                "double_stream_mean": None, "single_stream_mean": None}

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["block_kind", "block_idx"])["enrichment"].mean().reset_index().sort_values(
        "enrichment", ascending=False)
    peak_block = list(agg.iloc[0][["block_kind", "block_idx"]])
    trough_block = list(agg.iloc[-1][["block_kind", "block_idx"]])
    double_mean = df[df.block_kind == "double"]["enrichment"].mean()
    single_mean = df[df.block_kind == "single"]["enrichment"].mean()
    print(f"  Peak block: {peak_block}  double-mean={double_mean:.2f}x  single-mean={single_mean:.2f}x")

    return {"rows": rows, "peak_block": peak_block, "trough_block": trough_block,
            "double_stream_mean": double_mean, "single_stream_mean": single_mean}


def probe_block(pipe, gd_processor, gd_model, block_key, prompts, resolution: int | None = None):
    X, y_enrich = [], []
    size_kwargs = {"height": resolution, "width": resolution} if resolution else {}
    for prompt, word in prompts:
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1)
        text_seq_len = prompt_embeds.shape[1]
        with AttentionCapture(pipe.transformer, restrict_query_to=text_seq_len) as cap:
            image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=50, guidance_scale=4.0,
                         generator=torch.Generator("cuda").manual_seed(0), **size_kwargs).images[0]
        if block_key not in cap.captured:
            continue
        gd_score, gd_results = detect_score(gd_processor, gd_model, image, word)
        if gd_score == 0.0 or len(gd_results["boxes"]) == 0:
            continue
        bbox = gd_results["boxes"][gd_results["scores"].argmax()].tolist()

        attn = cap.captured[block_key]["attn"]
        key_t = cap.captured[block_key]["key"]
        image_seq_len = attn.shape[1] - text_seq_len
        grid_side = int(round(image_seq_len ** 0.5))
        if grid_side * grid_side != image_seq_len:
            continue
        target_mask = bbox_to_patch_mask(bbox, image.size, grid_side)
        image_keys = key_t[text_seq_len:]
        pooled_key = image_keys[target_mask].mean(dim=0) if target_mask.any() else image_keys.mean(dim=0)

        t2i = attn[:, text_seq_len:]
        attn_to_target = t2i[:, target_mask].mean().item() if target_mask.any() else float("nan")
        attn_to_all = t2i.mean().item()
        enrichment = attn_to_target / attn_to_all if attn_to_all > 0 else float("nan")

        X.append(pooled_key.numpy())
        y_enrich.append(enrichment)
        del cap
        gc.collect()
        torch.cuda.empty_cache()
    return np.array(X), np.array(y_enrich)


def run_key_probe(pipe, gd_processor, gd_model, peak_block, trough_block,
                   resolution: int | None = None) -> dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import r2_score

    out = {}
    for label, block_key in (("PEAK", peak_block), ("TROUGH", trough_block)):
        if block_key is None:
            out[label] = None
            continue
        block_key = tuple(block_key)
        X, yv = probe_block(pipe, gd_processor, gd_model, block_key, KEY_PROBE_PROMPTS, resolution)
        if len(X) < 4:
            print(f"  {label} block {block_key}: too few valid samples (n={len(X)})")
            out[label] = {"block": list(block_key), "n": len(X), "r2": None, "corr": None}
            continue
        loo = LeaveOneOut()
        preds = cross_val_predict(LinearRegression(), X, yv, cv=loo)
        r2 = r2_score(yv, preds)
        corr = float(np.corrcoef(yv, preds)[0, 1]) if len(set(preds)) > 1 else None
        print(f"  {label} block {block_key}: LOO R²={r2:.3f}  corr={corr}  n={len(X)}")
        out[label] = {"block": list(block_key), "n": len(X), "r2": r2, "corr": corr}
    return out


def main(step: int, checkpoint_dir: str | None, out_dir: Path, skip_images: bool, skip_b2: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    assert HF_TOKEN, "HF_TOKEN not set -- klein-Base-9B is gated."

    pipe = load_pipeline(checkpoint_dir)

    if not skip_images:
        print("\n-- Generating 20 comparison images --")
        generate_comparison_images(pipe, out_dir)
    else:
        print("\n-- Skipping 20 comparison images (--skip-images) --")

    print("\n-- Loading scoring models (Grounding DINO) --")
    gd_processor, gd_model = load_scoring_models()

    print("\n-- B1: block sweep --")
    block_sweep = run_block_sweep(pipe, gd_processor, gd_model)

    if not skip_b2:
        print("\n-- B2: key-vector probe --")
        key_probe = run_key_probe(pipe, gd_processor, gd_model,
                                   block_sweep["peak_block"], block_sweep["trough_block"])
    else:
        print("\n-- Skipping B2 (--skip-b2) --")
        key_probe = None

    results = {
        "step": step,
        "checkpoint_dir": checkpoint_dir,
        "block_sweep": block_sweep,
        "key_probe": key_probe,
    }
    (out_dir / "explain_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDone. Results -> {out_dir / 'explain_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, required=True, help="0 for pre-training baseline")
    p.add_argument("--checkpoint-dir", default=None, help="Omit for the base (no-LoRA) model")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--skip-images", action="store_true",
                    help="Skip the 20-image comparison set -- for cheap B1-only noise-floor reruns.")
    p.add_argument("--skip-b2", action="store_true",
                    help="Skip B2 (key-vector probe) -- for cheap B1-only noise-floor reruns.")
    args = p.parse_args()
    main(args.step, args.checkpoint_dir, args.out_dir, args.skip_images, args.skip_b2)
