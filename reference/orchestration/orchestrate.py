"""
Eval-only orchestrator — push images to pod, run evaluation, pull results.

Images are generated locally (or already on T drive) before running this.
Spins up a GPU pod (>=24GB recommended) for GeoReasoner + CLIPScore + StreetCLIP.

Three-pass design (prevents VRAM conflicts and keeps each model in its native venv):
  Pass 1 — GeoReasoner : geoeval venv, --skip-clip --skip-streetclip
  Pass 2 — CLIPScore   : main venv,    --skip-georeasoner --skip-streetclip
  Pass 3 — StreetCLIP  : geoeval venv, --skip-georeasoner --skip-clip

    python scripts/runpod/orchestrate.py            # full eval run
    python scripts/runpod/orchestrate.py --dry-run  # launch + nvidia-smi + terminate
    python scripts/runpod/orchestrate.py --pod-id X # reuse a running pod

SSH key: ~/.runpod/ssh/runpodctl-ssh-key  (registered via: runpodctl ssh add-key)
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import paramiko
import requests
import yaml

ROOT         = Path(__file__).resolve().parents[2]
cfg          = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
rp           = cfg["runpod"]
LOG_DIR      = Path(cfg["paths"]["logs"])
SSH_KEY      = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
REMOTE_DIR   = "/workspace/project"
GEOEVAL_VENV = "/workspace/.geoeval_venv"

MAX_RUNTIME_MIN = 90     # watchdog hard-kill — eval should finish well within this
COST_CEILING    = 5.00   # abort if estimated cost exceeds this
GPU_RATES       = {
    "T4":      0.200, "RTX 3080": 0.200, "RTX 3090": 0.220,
    "RTX A5000": 0.240, "RTX A6000": 0.330, "RTX 4090": 0.340,
}


# ── RunPod API ────────────────────────────────────────────────────────────────

def _api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY") or cfg["api_keys"].get("runpod", "")
    if not k:
        sys.exit("Set RUNPOD_API_KEY in ~/.bashrc")
    return k


def _gql(query: str, variables: dict = None) -> dict:
    resp = requests.post(
        rp["graphql_url"],
        params={"api_key": _api_key()},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(str(body["errors"]))
    return body["data"]


LAUNCH_MUT = """
mutation Launch($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id name desiredStatus
    machine { gpuDisplayName dataCenterId }
  }
}
"""

POLL_QUERY = """
query Pod($id: String!) {
  pod(input: { podId: $id }) {
    id desiredStatus costPerHr
    runtime {
      ports { ip privatePort publicPort isIpPublic }
      gpus  { gpuUtilPercent }
    }
  }
}
"""

TERM_MUT = "mutation { podTerminate(input: { podId: \"%s\" }) }"


def launch_pod() -> dict:
    """Launch cheapest >=16GB GPU (eval-only — Qwen-VL-Chat fits in 16GB fp16)."""
    sys.path.insert(0, str(ROOT))
    from scripts.runpod.pod_manager import available_gpus

    candidates = available_gpus(min_vram=16, secure_only=True)
    if not candidates:
        sys.exit("No 16GB+ GPUs available right now. Try again later.")

    est_cost = candidates[0]["price_hr"] * (MAX_RUNTIME_MIN / 60)
    if est_cost > COST_CEILING:
        sys.exit(f"Estimated cost ${est_cost:.2f} exceeds ceiling ${COST_CEILING:.2f}. Aborting.")

    # For eval-only, prefer budget GPUs — T4/RTX 3090 are ideal for Qwen-VL inference
    PREFERRED = ["T4", "RTX 3080", "RTX 3090", "A5000"]
    def _rank(g):
        for i, name in enumerate(PREFERRED):
            if name.lower() in g["name"].lower():
                return (i, g["price_hr"])
        return (len(PREFERRED), g["price_hr"])
    candidates.sort(key=_rank)

    print(f"Candidates (top 5, eval-preferred order):")
    for g in candidates[:5]:
        print(f"  {g['name']:<42} {g['vram_gb']}GB  ${g['price_hr']:.3f}/hr")

    for gpu in candidates:
        print(f"\nTrying {gpu['name']} (${gpu['price_hr']:.3f}/hr)...")
        pod_input = {
            "name":              "mycelium-eval",
            "imageName":         rp["docker_image"],
            "gpuTypeId":         gpu["id"],
            "gpuCount":          1,
            "cloudType":         "SECURE",
            "containerDiskInGb": 50,   # eval-only: Qwen-VL 19GB + LoRAs + system
            "volumeInGb":        0,
            "ports":             "8888/http,22/tcp",
            "startSsh":          True,
            "startJupyter":      True,
            "env": [
                {"key": "HF_TOKEN",      "value": os.environ.get("HF_TOKEN", "")},
                {"key": "RUNPOD_API_KEY","value": _api_key()},
                {"key": "JUPYTER_PASSWORD", "value": os.environ["JUPYTER_PASSWORD"]},
            ],
        }
        try:
            data    = _gql(LAUNCH_MUT, {"input": pod_input})
            pod     = data["podFindAndDeployOnDemand"]
            machine = pod.get("machine") or {}
            print(f"Pod launched: {pod['id']}  GPU={machine.get('gpuDisplayName')}  "
                  f"region={machine.get('dataCenterId')}")
            return {"id": pod["id"], "gpu": machine.get("gpuDisplayName", gpu["name"]),
                    "cost_hr": gpu["price_hr"]}
        except RuntimeError as e:
            err = str(e)
            if any(x in err.lower() for x in ["not enough", "unavailable", "no instance",
                                               "no longer any instance", "supply_constraint",
                                               "does not have the resources", "try a different machine"]):
                print("  Not available, trying next...")
                continue
            raise

    sys.exit("No suitable GPU available. Try again later.")


def wait_for_ssh(pod_id: str, timeout_s: int = 600) -> tuple[str, int]:
    print("Waiting for pod SSH to become ready", end="", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            pod   = _gql(POLL_QUERY, {"id": pod_id})["pod"]
            ports = (pod.get("runtime") or {}).get("ports") or []
            entry = next((p for p in ports if p["privatePort"] == 22 and p.get("isIpPublic")), None)
            if entry and entry.get("ip") and entry.get("publicPort"):
                print(f"  ready: {entry['ip']}:{entry['publicPort']}")
                return entry["ip"], int(entry["publicPort"])
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(15)
    raise RuntimeError(f"Pod {pod_id} did not expose SSH within {timeout_s}s")


def terminate(pod_id: str):
    for attempt in range(1, 4):
        try:
            _gql(TERM_MUT % pod_id)
            print(f"Pod {pod_id} terminated.")
            return
        except Exception as e:
            if attempt < 3:
                print(f"  terminate attempt {attempt} failed ({e}), retrying in 10s...")
                time.sleep(10)
            else:
                print(f"WARNING: terminate failed after 3 attempts: {e}  — check RunPod console!")


# ── SSH / SFTP ────────────────────────────────────────────────────────────────

def ssh_connect(host: str, port: int, retries: int = 5) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, retries + 1):
        try:
            client.connect(hostname=host, port=port, username="root",
                           key_filename=str(SSH_KEY), timeout=30)
            return client
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"SSH connect failed after {retries} attempts: {e}")
            print(f"  SSH not ready yet ({e}), retrying in 15s...")
            time.sleep(15)


def run_remote(client: paramiko.SSHClient, cmd: str, log_fh=None) -> int:
    transport = client.get_transport()
    chan      = transport.open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    chan.setblocking(False)
    while True:
        if chan.recv_ready():
            chunk = chan.recv(4096).decode("utf-8", errors="replace")
            print(chunk, end="", flush=True)
            if log_fh:
                log_fh.write(chunk)
        if chan.recv_stderr_ready():
            chunk = chan.recv_stderr(4096).decode("utf-8", errors="replace")
            print(chunk, end="", flush=True)
            if log_fh:
                log_fh.write(chunk)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.1)
    return chan.recv_exit_status()


def _sftp_mkdir(sftp, path: str):
    parts = path.lstrip("/").split("/")
    cur   = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            sftp.mkdir(cur)
        except IOError:
            pass


def push_dir(client: paramiko.SSHClient, local: Path, remote: str):
    sftp = client.open_sftp()
    _sftp_mkdir(sftp, remote)
    for item in local.rglob("*"):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        rel   = item.relative_to(local)
        rpath = f"{remote}/{rel}".replace("\\", "/")
        if item.is_dir():
            _sftp_mkdir(sftp, rpath)
        else:
            sftp.put(str(item), rpath)
    sftp.close()
    print(f"  Pushed {local} → {remote}")


def pull_dir(client: paramiko.SSHClient, remote: str, local: Path):
    import stat
    sftp = client.open_sftp()
    local.mkdir(parents=True, exist_ok=True)

    def _pull_recursive(rdir: str, ldir: Path):
        for entry in sftp.listdir_attr(rdir):
            rpath = f"{rdir}/{entry.filename}"
            lpath = ldir / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                lpath.mkdir(exist_ok=True)
                _pull_recursive(rpath, lpath)
            else:
                sftp.get(rpath, str(lpath))

    _pull_recursive(remote, local)
    sftp.close()
    print(f"  Pulled {remote} → {local}")


# ── Watchdog ──────────────────────────────────────────────────────────────────

def start_watchdog(pod_id: str, max_minutes: int) -> threading.Event:
    stop_event = threading.Event()

    def _watch():
        if not stop_event.wait(timeout=max_minutes * 60):
            print(f"\n[WATCHDOG] {max_minutes}min limit hit — terminating pod {pod_id}")
            terminate(pod_id)

    threading.Thread(target=_watch, daemon=True).start()
    return stop_event


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _pod_env() -> str:
    hf_token = os.environ.get("HF_TOKEN", "")
    return (
        f"export HF_TOKEN={hf_token}; "
        "export HF_HOME=/workspace/models; "
        "export HF_DATASETS_CACHE=/workspace/datasets; "
        "export TRANSFORMERS_CACHE=/workspace/models/hub; "
        "export HF_HUB_DISABLE_XET=1; "
    )


def step_push_scripts(client):
    print("\n── STEP: push scripts + configs ──")
    push_dir(client, ROOT / "scripts", f"{REMOTE_DIR}/scripts")
    push_dir(client, ROOT / "configs", f"{REMOTE_DIR}/configs")
    sftp = client.open_sftp()
    sftp.put(str(ROOT / "scripts/runpod/pod_init.sh"), "/workspace/pod_init.sh")
    sftp.close()
    print("  Uploaded pod_init.sh → /workspace/pod_init.sh")


def step_init(client, log_fh):
    print(f"\n── STEP: pod_init.sh (eval-only: Qwen-VL + LoRAs) ──")
    rc = run_remote(client, f"{_pod_env()}bash /workspace/pod_init.sh", log_fh)
    if rc != 0:
        raise RuntimeError(f"pod_init.sh failed (exit {rc})")


def step_push_images(client):
    """Push locally-generated FAL images to pod's /workspace/outputs/images/."""
    local_images = Path(cfg["paths"]["results"]) / "w1_outputs" / "images"
    if not local_images.exists():
        raise RuntimeError(
            f"Images directory not found: {local_images}\n"
            "Run generate_via_api.py first."
        )
    pngs = sorted(local_images.glob("*.png"))
    if not pngs:
        raise RuntimeError(
            f"No .png images found in {local_images}\n"
            "Run generate_via_api.py first."
        )

    print(f"\n── STEP: push {len(pngs)} images to pod ──")
    sftp = client.open_sftp()
    _sftp_mkdir(sftp, "/workspace/outputs/images")
    for img in pngs:
        sftp.put(str(img), f"/workspace/outputs/images/{img.name}")
    sftp.close()
    print(f"  Pushed {len(pngs)} images → /workspace/outputs/images/")


def step_smoke_geoeval(client, log_fh):
    print(f"\n── STEP: geoeval venv smoke test ──")
    rc = run_remote(
        client,
        f"{_pod_env()}{GEOEVAL_VENV}/bin/python -c \""
        "from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor; "
        "from peft import PeftModel; "
        "import numpy; "
        "print('imports ok — numpy:', numpy.__version__)\"",
        log_fh,
    )
    if rc != 0:
        raise RuntimeError("geoeval smoke test failed — dependency conflict, check pod log")


def step_eval_georeasoner(client, log_fh):
    print(f"\n── STEP: Pass 1 — GeoReasoner eval ({GEOEVAL_VENV}) ──")
    rc = run_remote(
        client,
        f"{_pod_env()}{GEOEVAL_VENV}/bin/python "
        f"{REMOTE_DIR}/scripts/eval/run_eval.py --skip-clip --skip-streetclip",
        log_fh,
    )
    if rc != 0:
        raise RuntimeError(f"GeoReasoner eval failed (exit {rc})")


def step_eval_clip(client, log_fh):
    print(f"\n── STEP: Pass 2 — CLIPScore eval (main venv) ──")
    rc = run_remote(
        client,
        f"{_pod_env()}source /workspace/.venv/bin/activate && "
        f"cd {REMOTE_DIR} && python scripts/eval/run_eval.py --skip-georeasoner --skip-streetclip",
        log_fh,
    )
    if rc != 0:
        raise RuntimeError(f"CLIPScore eval failed (exit {rc})")


def step_eval_streetclip(client, log_fh):
    print(f"\n── STEP: Pass 3 — StreetCLIP eval ({GEOEVAL_VENV}) ──")
    rc = run_remote(
        client,
        f"{_pod_env()}{GEOEVAL_VENV}/bin/python "
        f"{REMOTE_DIR}/scripts/eval/run_eval.py --skip-georeasoner --skip-clip",
        log_fh,
    )
    if rc != 0:
        raise RuntimeError(f"StreetCLIP eval failed (exit {rc})")


def step_pull(client):
    print("\n── STEP: pull results ──")
    local_results = Path(cfg["paths"]["results"]) / "e02_corrected_eval"
    pull_dir(client, "/workspace/outputs", local_results)


def step_verify(expected_images: int = 60) -> bool:
    out  = Path(cfg["paths"]["results"]) / "w1_outputs"
    imgs = list((out / "images").glob("*.png")) if (out / "images").exists() else []
    csv  = out / "results.csv"
    print(f"\n── VERIFY ──")
    print(f"  Images : {len(imgs)} (expected {expected_images})")
    print(f"  CSV    : {'✓ exists' if csv.exists() else '✗ MISSING'}")
    ok = True
    if csv.exists():
        rows = len(csv.read_text().strip().splitlines()) - 1
        print(f"  CSV rows: {rows} (expected {expected_images})")
        if rows < expected_images:
            print("  WARNING: fewer CSV rows than expected")
            ok = False
    else:
        ok = False
    if len(imgs) < expected_images:
        print(f"  WARNING: only {len(imgs)}/{expected_images} images present")
        ok = False
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, reuse_pod_id: str = ""):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode     = "dryrun" if dry_run else "eval"
    log_path = LOG_DIR / f"w1_{mode}_{ts}.log"

    pod_id   = None
    watchdog = None
    t_start  = time.time()

    with open(log_path, "w") as log_fh:
        log_fh.write(f"W1 eval orchestration ({mode}) — {datetime.now()}\n\n")
        print(f"Log → {log_path}\n")

        try:
            print("="*60)
            print(f" W1 EVAL PIPELINE — {mode.upper()}")
            print("="*60)

            if reuse_pod_id:
                pod_id  = reuse_pod_id
                cost_hr = 0.0
                print(f"Reusing existing pod {pod_id}")
                host, port = wait_for_ssh(pod_id, timeout_s=120)
            else:
                pod     = launch_pod()
                pod_id  = pod["id"]
                cost_hr = pod["cost_hr"]
                watchdog = start_watchdog(pod_id, MAX_RUNTIME_MIN)
                log_fh.write(f"Pod: {pod_id}  GPU: {pod['gpu']}  ${cost_hr:.3f}/hr\n\n")
                host, port = wait_for_ssh(pod_id)

            print("Connecting via SSH...")
            client = ssh_connect(host, port)
            print(f"Connected to {host}:{port}")

            if dry_run:
                print("\n── DRY RUN: nvidia-smi ──")
                run_remote(client, "nvidia-smi", log_fh)
                print("\nDry run complete.")
                return

            # 1. Push scripts (pod_init.sh must be on pod before we run it)
            step_push_scripts(client)

            # 2. Run pod_init.sh — installs deps, downloads Qwen-VL + LoRAs
            step_init(client, log_fh)

            # 3. Smoke-test geoeval imports before wasting time pushing 60 images
            step_smoke_geoeval(client, log_fh)

            # 4. Push locally-generated images to pod
            step_push_images(client)

            # 5. Pass 1 — GeoReasoner (geoeval venv, transformers>=4.36,<4.40)
            step_eval_georeasoner(client, log_fh)

            # 6. Pass 2 — CLIPScore (main venv, open-clip-torch)
            step_eval_clip(client, log_fh)

            # 7. Pass 3 — StreetCLIP (geoeval venv, transformers CLIPModel)
            #    Three-pass design keeps each model in its native venv and
            #    prevents GeoReasoner (~19GB) + StreetCLIP from sharing VRAM.
            step_eval_streetclip(client, log_fh)

            # 8. Pull results.csv (images already local — no need to re-pull)
            step_pull(client)

            # 9. Verify
            step_verify(expected_images=60)

            elapsed = (time.time() - t_start) / 3600
            cost    = elapsed * cost_hr
            print(f"\n{'='*60}")
            print(f" DONE  —  {elapsed*60:.0f} min  ≈  ${cost:.2f}")
            print(f" Results → {Path(cfg['paths']['results']) / 'e02_corrected_eval' / 'results.csv'}")
            print(f"{'='*60}")
            log_fh.write(f"\nElapsed: {elapsed*60:.1f} min  Estimated cost: ${cost:.2f}\n")

        finally:
            if watchdog:
                watchdog.set()
            if pod_id and not reuse_pod_id:
                print(f"\nTearing down pod {pod_id}...")
                terminate(pod_id)
            elif reuse_pod_id:
                print(f"\nPod {pod_id} preserved (--pod-id reuse mode).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="W1 eval orchestrator (eval-only, images pushed from local)")
    p.add_argument("--dry-run", action="store_true",
                   help="Launch pod, run nvidia-smi, terminate (~$0.05)")
    p.add_argument("--pod-id", default="",
                   help="Reuse a running pod instead of launching a new one")
    args = p.parse_args()
    main(dry_run=args.dry_run, reuse_pod_id=args.pod_id)
