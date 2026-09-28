"""
E06 pod-side training script. Adapted from Work/scripts/generate/parallel_finetune_train.py
(R4 -- same overfit-check/full-run structure, loss parsing, result-file contract reused
near-verbatim) with two real changes:
  1. No trigger word -- --instance-prompt is a required CLI arg (the training script itself
     still needs a non-empty value, B43), not a hardcoded TRIGGER_WORD. E06 passes each cell's
     own canonical prompt here (see orchestrate_training.py), not an invented token.
  2. Single fixed resolution (320) instead of per-branch resolution -- E06 has no real/AI
     split at the training-recipe level (both halves of each cell's 40 images train together
     at 320, per README.md SS2/SS3), unlike the 07-31 experiment's real_512/ai_768 branches.

Everything else (rank 16, LR 1e-4, guidance_scale=1, overfit-check mode, result-file schema)
is identical to the already-validated recipe -- not re-derived.

Pod-side. Requires the diffusers-from-source clone + torchao/bitsandbytes/datasets/prodigyopt
(same bootstrap as parallel_finetune_train.py) and HF_TOKEN set.

    python train_lora.py --dataset-dir /path/to/training_data/india_generic_rural_road \
        --instance-prompt "A rural road in India: ..." \
        --output-dir /workspace/outputs/india_generic_rural_road --overfit-check
    python train_lora.py --dataset-dir /path/to/training_data/india_generic_rural_road \
        --instance-prompt "A rural road in India: ..." \
        --output-dir /workspace/outputs/india_generic_rural_road \
        --max-train-steps 800 --checkpointing-steps 100
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TRAIN_SCRIPT = Path("/workspace/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py")
RESOLUTION = 320  # LOCKED, README.md SS3 -- global min dimension across all 160 kept training
                   # crops is 328px; 320 is the largest multiple of 64 that fits with zero upscale.


def stage_subset(dataset_dir: Path, n_images: int | None) -> Path:
    """For --overfit-check: stages N images + their metadata rows into a temp dir, so the real
    dataset_dir is untouched and reusable for the full run afterward.

    BUG FIX (2026-08-18, Sol review): taking the first N rows unconditionally always picked
    real images -- prep_training_data.py writes all 20 real rows before all 20 AI rows (R4:
    ai_source loop runs after the direct-real loop), so the overfit-check hard gate never
    actually exercised an AI-generated input (decode/caption/cache/learn), even though AI
    images are half the full training recipe. Split the subset evenly between real and
    ai_-prefixed rows (prep_training_data.py's own naming convention) so the gate tests both."""
    if n_images is None:
        return dataset_dir

    rows = [json.loads(line) for line in (dataset_dir / "metadata.jsonl").read_text().splitlines()]
    ai_rows = [r for r in rows if r["file_name"].startswith("ai_")]
    real_rows = [r for r in rows if not r["file_name"].startswith("ai_")]
    n_ai = n_images // 2
    n_real = n_images - n_ai
    subset_rows = real_rows[:n_real] + ai_rows[:n_ai]
    assert len(subset_rows) == n_images, (
        f"overfit-check subset came up short: wanted {n_real} real + {n_ai} ai, "
        f"got {len(real_rows[:n_real])} real + {len(ai_rows[:n_ai])} ai from "
        f"{len(real_rows)} real / {len(ai_rows)} ai rows available")

    staging = Path(tempfile.mkdtemp(prefix="overfit_check_"))
    for row in subset_rows:
        shutil.copy(dataset_dir / row["file_name"], staging / row["file_name"])
    with open(staging / "metadata.jsonl", "w") as f:
        for row in subset_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Staged {len(subset_rows)} images for overfit-check -> {staging}")
    return staging


def parse_losses(log_text: str) -> dict:
    raw = re.findall(r"loss=([\d.eE+-]+)", log_text)
    losses = []
    for m in raw:
        try:
            v = float(m)
        except ValueError:
            continue
        if v == v and v not in (float("inf"), float("-inf")):  # finite, not NaN
            losses.append(v)
    if not losses:
        return {"n": 0, "first_third_mean": None, "last_third_mean": None, "all": []}
    third = max(1, len(losses) // 3)
    return {
        "n": len(losses),
        "first_third_mean": sum(losses[:third]) / third,
        "last_third_mean": sum(losses[-third:]) / third,
        "all": losses,
    }


def main(dataset_dir: Path, instance_prompt: str, output_dir: Path, max_train_steps: int,
         checkpointing_steps: int, overfit_check: bool, overfit_n_images: int, log_out: Path):
    train_dataset_dir = stage_subset(dataset_dir, overfit_n_images if overfit_check else None)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate", "launch", str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.2-klein-base-9B",
        "--dataset_name", str(train_dataset_dir), "--caption_column", "text",
        "--instance_prompt", instance_prompt,
        "--output_dir", str(output_dir),
        "--do_fp8_training", "--gradient_checkpointing", "--cache_latents",
        "--resolution", str(RESOLUTION),
        "--train_batch_size", "1", "--guidance_scale", "1",
        "--use_8bit_adam", "--gradient_accumulation_steps", "4",
        "--optimizer", "adamW", "--learning_rate", "1e-4", "--rank", "16",
        "--lr_scheduler", "constant", "--lr_warmup_steps", "20",
        "--max_train_steps", str(max_train_steps),
        "--checkpointing_steps", str(checkpointing_steps),
        "--seed", "42",
    ]
    print(f"mode={'overfit-check' if overfit_check else 'full-run'}  resolution={RESOLUTION}  "
          f"max_train_steps={max_train_steps}  checkpointing_steps={checkpointing_steps}")
    print("Launching:\n", " ".join(cmd))

    log_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
        log_lines.append(line)
    process.wait()

    log_text = "".join(log_lines)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.write_text(log_text)
    loss_summary = parse_losses(log_text)
    result = {"mode": "overfit_check" if overfit_check else "full_run", "resolution": RESOLUTION,
              "max_train_steps": max_train_steps, "checkpointing_steps": checkpointing_steps,
              "return_code": process.returncode, "loss_summary": loss_summary}
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2))

    if overfit_check and train_dataset_dir != dataset_dir:
        shutil.rmtree(train_dataset_dir, ignore_errors=True)

    if process.returncode != 0:
        print(f"\nTraining exited with code {process.returncode}")
        raise SystemExit(process.returncode)

    ls = loss_summary
    print(f"\nDone. n={ls['n']}  first-third mean={ls['first_third_mean']}  "
          f"last-third mean={ls['last_third_mean']}")
    if overfit_check and ls["first_third_mean"] and ls["last_third_mean"]:
        dropped = ls["last_third_mean"] < ls["first_third_mean"] * 0.9
        print(f"Overfit-check verdict: {'PASS -- loss dropped meaningfully' if dropped else 'FAIL -- loss did not meaningfully drop, do not proceed to full run'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--instance-prompt", required=True,
                   help="Cell's own canonical prompt (paraphrase 0) -- no invented trigger token.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-train-steps", type=int, default=800)
    p.add_argument("--checkpointing-steps", type=int, default=100)
    p.add_argument("--overfit-check", action="store_true")
    p.add_argument("--overfit-n-images", type=int, default=4)
    p.add_argument("--log-out", type=Path, default=None)
    args = p.parse_args()

    if args.log_out is None:
        args.log_out = args.output_dir / "train_log.txt"
    if args.overfit_check:
        args.max_train_steps = min(args.max_train_steps, 200)
        args.checkpointing_steps = args.max_train_steps + 1  # no intermediate saves needed

    main(args.dataset_dir, args.instance_prompt, args.output_dir, args.max_train_steps,
         args.checkpointing_steps, args.overfit_check, args.overfit_n_images, args.log_out)
