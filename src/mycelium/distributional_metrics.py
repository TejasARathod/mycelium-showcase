"""
E04 — Distributional comparison metrics: real photos vs. generated images.

Answers "does the generated set actually cover the diversity of the real
set, or does it collapse to a narrow slice of it" — the question GeoDiv's
Hill Number alone doesn't fully answer, since it never compares against a
real-image reference.

Metrics (all computed on image embeddings, not raw pixels):
  - KID   : Kernel Inception Distance == MMD with a degree-3 polynomial
            kernel (Binkowski et al. 2018). Same underlying statistic as
            the Galashov DMMD paper, applied here as an evaluation metric
            instead of a training signal.
  - Precision / Recall (generative) : Kynkaanniemi et al. 2019. Precision =
            fraction of generated images that fall inside the real-image
            manifold (plausibility). Recall = fraction of the real
            manifold's variety that the generated set actually covers
            (diversity). These are reported separately on purpose --
            averaging them (as FID does) would hide exactly the
            high-precision/low-recall collapse pattern we're testing for.
  - Vendi Score : Friedman & Dieng 2023. A single diversity number per set
            (not comparative), based on the effective rank of the
            pairwise-similarity kernel matrix. Same mathematical family as
            MMD (kernel eigenvalues) -- reported per-set so it can be
            plotted as generated-vs-real bars.

Feature extractor: CLIP ViT-L/14 by default (matches the project's existing
CLIPScorer backbone, so results sit in the same embedding space as
CLIPScore/StreetCLIP). DINOv2 is offered as a second, non-text-supervised
backbone -- Hall et al. found it correlates better with human similarity
judgment than CLIP and isn't subject to CLIP's text-conditioned bias, so
running both is the honest way to check whether a finding is backbone-
specific or real.

Usage:
    from scripts.eval.distributional_metrics import DistributionalMetrics
    dm = DistributionalMetrics(backbone="clip")
    real_feats = dm.extract_embeddings(real_image_paths)
    gen_feats  = dm.extract_embeddings(gen_image_paths)
    result = dm.compare(real_feats, gen_feats)
    # result = {"kid": ..., "precision": ..., "recall": ...,
    #           "vendi_real": ..., "vendi_generated": ...}
"""

from pathlib import Path

import numpy as np
import torch


class DistributionalMetrics:
    def __init__(self, backbone: str = "clip", device: str = "cuda"):
        self.backbone = backbone
        self.device = device
        if backbone == "clip":
            import open_clip
            # "ViT-L-14" (no suffix) loads with standard GELU; OpenAI's actual "openai"
            # checkpoint was trained with QuickGELU -- open_clip warns about this
            # mismatch at load time. "-quickgelu" is the correct variant name for
            # numerical fidelity to the real OpenAI weights (verified against the
            # official jmhessel/clipscore reference implementation, which gets this
            # right for free by using OpenAI's own `clip` package instead of open_clip).
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14-quickgelu", pretrained="openai"
            )
            self.model = self.model.to(device).eval()
        elif backbone == "dinov2":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    @torch.no_grad()
    def extract_embeddings(self, image_paths: list, verbose: bool = True) -> np.ndarray:
        """Returns (N, D) L2-normalized embedding matrix."""
        from PIL import Image

        feats = []
        n = len(image_paths)
        for i, path in enumerate(image_paths, 1):
            if verbose:
                print(f"  [{i:>3}/{n}] {Path(path).name}", end="\r", flush=True)
            img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            if self.backbone == "clip":
                f = self.model.encode_image(img)
            else:
                f = self.model(img)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.squeeze(0).cpu().numpy())
        if verbose:
            print()
        return np.stack(feats)

    # ── KID (== MMD, polynomial kernel, degree 3) ───────────────────────────

    @staticmethod
    def _poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
        # Binkowski et al. 2018 divide by feature dim d because raw Inception
        # pool3 features are unnormalized and their dot products scale with d.
        # This class's embeddings are always L2-normalized (see
        # extract_embeddings), so x @ y.T is already a bounded cosine
        # similarity in [-1, 1] -- dividing by d (e.g. 768 for CLIP ViT-L/14)
        # crushes it to ~0, making the kernel collapse to ~1 regardless of
        # actual similarity and destroying KID's sensitivity. No /d for
        # normalized inputs.
        return (x @ y.T + 1.0) ** degree

    def compute_kid(self, real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
        """Unbiased MMD^2 estimator (Binkowski et al. 2018 Eq. 2)."""
        m, n = len(real_feats), len(gen_feats)
        k_rr = self._poly_kernel(real_feats, real_feats)
        k_gg = self._poly_kernel(gen_feats, gen_feats)
        k_rg = self._poly_kernel(real_feats, gen_feats)

        # Exclude diagonal (self-similarity) from the within-set terms.
        sum_rr = (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
        sum_gg = (k_gg.sum() - np.trace(k_gg)) / (n * (n - 1))
        sum_rg = k_rg.sum() / (m * n)

        return float(sum_rr + sum_gg - 2 * sum_rg)

    # ── Precision / Recall (Kynkaanniemi et al. 2019) ───────────────────────

    @staticmethod
    def _knn_radii(feats: np.ndarray, k: int = 3) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)  # +1: excludes self
        dists, _ = nn.kneighbors(feats)
        return dists[:, -1]  # distance to k-th nearest neighbour (self excluded)

    def compute_precision_recall(self, real_feats: np.ndarray, gen_feats: np.ndarray,
                                   k: int = 3) -> dict:
        """
        Precision: fraction of generated samples inside the real manifold
                   (plausibility -- "does it look like it belongs").
        Recall:    fraction of real samples inside the generated manifold
                   (coverage -- "does the generated set span the real variety").
        Manifold membership = falls within another point's k-NN radius ball.
        """
        real_radii = self._knn_radii(real_feats, k)
        gen_radii = self._knn_radii(gen_feats, k)

        def _in_manifold(query_feats, ref_feats, ref_radii):
            hits = 0
            for q in query_feats:
                d = np.linalg.norm(ref_feats - q, axis=1)
                if np.any(d <= ref_radii):
                    hits += 1
            return hits / len(query_feats)

        precision = _in_manifold(gen_feats, real_feats, real_radii)
        recall = _in_manifold(real_feats, gen_feats, gen_radii)
        return {"precision": precision, "recall": recall}

    # ── Density / Coverage (Naeem et al. 2020, ICML) ────────────────────────
    # Formulas verified directly against the primary PDF (Sources/05-07/naeem20a.pdf,
    # 2026-07-06): Density = Eq. 4, Coverage = Eq. 5, analytic E[coverage] under
    # identical real/fake distributions = Eq. 9 (Lemma 2). Reference implementation:
    # github.com/clovaai/generative-evaluation-prdc (cited in the paper, not
    # independently re-verified against this repo -- flagged below).
    #
    # Density/Coverage were proposed specifically to fix two failure modes the
    # paper documents in Precision/Recall (Kynkaanniemi et al. 2019, already used
    # above): precision is fooled by real-set outliers (an outlier's oversized
    # k-NN ball inflates precision for unrelated fakes), and the fake-built
    # manifold recall uses makes recall unreliable for diverse-but-unrealistic
    # generators. Both replacements build their balls around the REAL set only.

    def compute_density_coverage(self, real_feats: np.ndarray, gen_feats: np.ndarray,
                                   k: int = 3) -> dict:
        """
        Density (Eq. 4): for every real point X_i with k-NN radius r_i, count how
        many generated points fall inside B(X_i, r_i); sum over all real points,
        divide by (k * n_generated). Not upper-bounded by 1 -- can exceed it in
        regions where reals are packed tighter than their own k-NN radius implies.

        Coverage (Eq. 5): fraction of real points whose ball B(X_i, r_i) contains
        at least one generated point. Bounded [0, 1]. This is Recall's replacement
        -- same "does the generated set span the real variety" question, but the
        manifold is built from the (less outlier-prone) real set instead of the
        fake set.
        """
        real_radii = self._knn_radii(real_feats, k)

        density_hits = 0
        coverage_hits = 0
        for x_i, r_i in zip(real_feats, real_radii):
            d = np.linalg.norm(gen_feats - x_i, axis=1)
            in_ball = d <= r_i
            density_hits += int(in_ball.sum())
            if in_ball.any():
                coverage_hits += 1

        density = density_hits / (k * len(gen_feats))
        coverage = coverage_hits / len(real_feats)
        return {"density": float(density), "coverage": float(coverage)}

    @staticmethod
    def expected_coverage_identical(n: int, m: int, k: int) -> float:
        """
        Analytic E[coverage] when real and fake are drawn from the *same*
        distribution (Naeem et al. 2020, Lemma 2, Eq. 9):
            E[coverage] = 1 - prod_{j=0}^{k-1}(n-1-j) / prod_{j=0}^{k-1}(m+n-1-j)
        As n=m -> infinity, E[coverage] -> 1 - 2^-k.
        Use this to sanity-check a chosen k *before* trusting a coverage number:
        the paper's own recommendation is to pick k so this expected value is
        comfortably high (e.g. > 0.95) at the actual n, m in hand -- otherwise a
        low coverage reading could just be an artifact of a too-small k, not a
        real diversity deficit.
        """
        num = 1.0
        den = 1.0
        for j in range(k):
            num *= (n - 1 - j)
            den *= (m + n - 1 - j)
        return float(1.0 - num / den)

    # ── Vendi Score (Friedman & Dieng 2023) ─────────────────────────────────

    def compute_vendi_score(self, feats: np.ndarray) -> float:
        """Effective diversity via entropy of the similarity-kernel eigenvalues."""
        n = len(feats)
        k = feats @ feats.T  # cosine similarity, since feats are L2-normalized
        k = k / n
        eigvals = np.linalg.eigvalsh(k)
        eigvals = eigvals[eigvals > 1e-12]  # drop numerical-noise negatives/zeros
        entropy = -np.sum(eigvals * np.log(eigvals))
        return float(np.exp(entropy))

    # ── Convenience: run everything at once ─────────────────────────────────

    def compare(self, real_feats: np.ndarray, gen_feats: np.ndarray, k: int = 3) -> dict:
        pr = self.compute_precision_recall(real_feats, gen_feats, k=k)
        dc = self.compute_density_coverage(real_feats, gen_feats, k=k)
        n, m = len(real_feats), len(gen_feats)
        return {
            "backbone": self.backbone,
            "n_real": n,
            "n_generated": m,
            "k": k,
            "kid": self.compute_kid(real_feats, gen_feats),
            "precision": pr["precision"],
            "recall": pr["recall"],
            "density": dc["density"],
            "coverage": dc["coverage"],
            "expected_coverage_at_k_if_identical": self.expected_coverage_identical(n, m, k),
            "vendi_real": self.compute_vendi_score(real_feats),
            "vendi_generated": self.compute_vendi_score(gen_feats),
        }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", required=True)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--backbone", default="clip", choices=["clip", "dinov2"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out-dir", default=None,
                   help="If set, saves result JSON + real/gen embedding .npy files here "
                        "(consumed by scripts/eval/plot_e04.py for the embedding scatter)")
    args = p.parse_args()

    real_paths = sorted(Path(args.real_dir).glob("*.jpg")) + sorted(Path(args.real_dir).glob("*.png"))
    gen_paths = sorted(Path(args.gen_dir).glob("*.jpg")) + sorted(Path(args.gen_dir).glob("*.png"))

    dm = DistributionalMetrics(backbone=args.backbone, device=args.device)
    print(f"Extracting {args.backbone} embeddings for {len(real_paths)} real images...")
    real_feats = dm.extract_embeddings(real_paths)
    print(f"Extracting {args.backbone} embeddings for {len(gen_paths)} generated images...")
    gen_feats = dm.extract_embeddings(gen_paths)

    result = dm.compare(real_feats, gen_feats, k=args.k)
    print(json.dumps(result, indent=2))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"distributional_metrics_{args.backbone}.json").write_text(json.dumps(result, indent=2))
        np.save(out_dir / f"real_embeddings_{args.backbone}.npy", real_feats)
        np.save(out_dir / f"gen_embeddings_{args.backbone}.npy", gen_feats)
        # Filename manifest, same row order as the .npy arrays above -- so downstream
        # scripts (e.g. seed_mode_decomposition.py) can parse each row's actual
        # condition from its filename instead of re-deriving row order by re-running
        # this same sorted-glob and assuming a fixed grid shape. That assumption is
        # fragile in general (e.g. E05's UK seeds run 42-106, crossing from 2-digit
        # to 3-digit -- lexicographic glob order stops matching numeric seed order
        # right at that boundary) -- an explicit manifest sidesteps the problem
        # entirely rather than relying on sort behavior staying accidentally correct.
        (out_dir / f"real_filenames_{args.backbone}.json").write_text(
            json.dumps([p.name for p in real_paths], indent=2))
        (out_dir / f"gen_filenames_{args.backbone}.json").write_text(
            json.dumps([p.name for p in gen_paths], indent=2))
        print(f"\nSaved -> {out_dir}")
