"""
smoke_test.py
=============

Runs the whole pipeline on synthetic data in a couple of minutes, with no
dataset and no GPU, and asserts on the things that must be true.

    python scripts/smoke_test.py            # everything
    python scripts/smoke_test.py --quick    # skip the training step

What it checks, and why each check exists
------------------------------------------
 1. TSC controlled test — the classes have matched average spatial spectra, so
    a per-frame frequency detector must fail while TSC succeeds. If this ever
    breaks, the novelty claim is broken, not the code style.
 2. Shapes and gradients — every branch produces the declared shape and
    receives gradient. Catches the classic "the gate collapsed and three
    branches are dead" bug at build time rather than after 6 hours of training.
 3. Identity-disjoint splitting — asserts no identity group spans two splits,
    and that a random split *does* leak. That contrast is the number the report
    quotes.
 4. End-to-end — synthetic videos -> manifest -> preprocess -> train 2 epochs
    -> evaluate. Catches the boring integration bugs.
 5. Evidence + template explanation — the reasoning layer degrades to the
    template backend with no VLM installed, and still produces something.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ==========================================================================

def test_tsc_controlled() -> bool:
    from fsvnet.synthetic import controlled_tsc_test
    print("\n1. TSC controlled experiment (variance-matched classes)")
    r = controlled_tsc_test(n=20, T=32, size=224, seed=3, verbose=True)
    ok1 = check("TSC separates temporal-only differences",
                r["tsc_best_auc"] > 0.90, f"AUC {r['tsc_best_auc']:.3f} (need >0.90)")
    ctrl = max(abs(r["control_marginal_hf_auc"] - 0.5),
               abs(r["control_marginal_slope_auc"] - 0.5))
    ok2 = check("per-frame spectral control stays near chance",
                ctrl < 0.20, f"|AUC-0.5| = {ctrl:.3f} (need <0.20)")
    return ok1 and ok2


def test_model_shapes() -> bool:
    from fsvnet.config import Config
    from fsvnet.model import build_model
    print("\n2. Model shapes, gradients and parameter budget")

    cfg = Config()
    cfg.device = "cpu"
    cfg.data.num_frames = 8
    cfg.model.pretrained = False          # no download in the test
    model = build_model(cfg)

    B, T, S = 2, cfg.data.num_frames, cfg.data.img_size
    clip = torch.randn(B, T, 3, S, S)
    mask = torch.ones(B, T, dtype=torch.bool)

    logits, info = model(clip, mask, return_evidence=True)
    ok = check("logits shape", tuple(logits.shape) == (B, 2), str(tuple(logits.shape)))
    ok &= check("branch attention shape",
                tuple(info["branch_weights"].shape) == (B, 4),
                str(tuple(info["branch_weights"].shape)))
    ok &= check("branch weights sum to 1",
                bool(torch.allclose(info["branch_weights"].sum(1),
                                    torch.ones(B), atol=1e-4)))
    ev = info["evidence"]
    F_exp = T // 2 + 1
    ok &= check("TSC map shape (B, T/2+1, K)",
                tuple(ev["tsc_tsc_map"].shape) == (B, F_exp, cfg.model.azimuthal_bins),
                str(tuple(ev["tsc_tsc_map"].shape)))
    ok &= check("SRM energy trajectory shape",
                tuple(ev["srm_energy"].shape) == (B, T))

    # every branch must receive gradient
    logits.sum().backward()
    grads = {}
    for name in model.branch_names:
        mod = getattr(model, name)
        gs = [p.grad.abs().sum().item() for p in mod.parameters()
              if p.requires_grad and p.grad is not None]
        grads[name] = float(sum(gs))
    ok &= check("all branches receive gradient",
                all(v > 0 for v in grads.values()),
                ", ".join(f"{k}={v:.2e}" for k, v in grads.items()))

    rep = model.param_report()
    print(f"      params: total {rep['TOTAL']['total']:,}  "
          f"trainable {rep['TOTAL']['trainable']:,}")
    print(f"      TSC branch adds {rep['tsc']['total']:,} params "
          f"({rep['tsc']['total'] / rep['TOTAL']['total'] * 100:.1f}% of the model)")
    return ok


def test_splits() -> bool:
    import pandas as pd

    from fsvnet.manifest import _group_overlap, make_splits
    print("\n3. Identity-disjoint splitting")

    rows = []
    for gid in range(30):
        for k in range(4):
            rows.append(dict(video_path=f"/x/id{gid}_{k}.mp4",
                             label=k % 2, label_name="fake" if k % 2 else "real",
                             identity=f"id{gid}", group=f"id{gid}",
                             dataset="synthetic", method="m", split=""))
    df = pd.DataFrame(rows)

    ident = make_splits(df.copy(), "identity", seed=1)
    ok = check("identity split has no group overlap", not _group_overlap(ident))
    ok &= check("identity split fills all three splits",
                set(ident.split.unique()) == {"train", "val", "test"})

    rnd = make_splits(df.copy(), "random", seed=1)
    leaked = bool(_group_overlap(rnd))
    ok &= check("random split DOES leak (this is the control)", leaked,
                "as expected" if leaked else "unexpectedly clean")
    return ok


def test_end_to_end(quick: bool = False) -> bool:
    from fsvnet.config import Config
    from fsvnet.dataset import make_loaders, preprocess_dataset
    from fsvnet.manifest import build_manifest, make_splits
    from fsvnet.model import build_model
    print("\n4. End-to-end: synthetic videos -> manifest -> cache -> train -> eval")

    tmp = Path(tempfile.mkdtemp(prefix="fsvnet_smoke_"))
    try:
        cfg = Config()
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.paths.data_dir = tmp / "data"
        cfg.paths.clip_cache = tmp / "clips"
        cfg.paths.ckpt_dir = tmp / "ckpt"
        cfg.paths.out_dir = tmp / "out"
        cfg.paths.pred_dir = tmp / "out" / "pred"
        cfg.paths.plot_dir = tmp / "out" / "plots"
        cfg.paths.evidence_dir = tmp / "out" / "evidence"
        cfg.paths.mkdirs()

        cfg.data.num_frames = 8
        cfg.data.frame_stride = 1
        cfg.data.img_size = 112
        cfg.data.face_backend = "none"
        cfg.data.min_faces_per_clip = 0.0
        cfg.data.clips_per_video_eval = 1
        cfg.model.pretrained = False
        cfg.model.freeze_spatial_epochs = 0
        cfg.train.epochs = 2
        cfg.train.batch_size = 2
        cfg.train.grad_accum = 2
        cfg.train.num_workers = 0
        cfg.train.amp = False
        cfg.run_name = "smoke"

        from fsvnet.synthetic import build_synthetic_dataset
        root = build_synthetic_dataset(cfg, n_per_class=12, T=20, size=128)

        df = build_manifest({"synthetic": root})
        ok = check("manifest built", len(df) == 24, f"{len(df)} rows")

        df = make_splits(df, "identity", (0.5, 0.25, 0.25), seed=0)
        df = preprocess_dataset(df, cfg)
        ok &= check("preprocessing cached every clip",
                    bool(df["ok"].all()), f"{int(df['ok'].sum())}/{len(df)}")

        loaders = make_loaders(df, cfg)
        ok &= check("loaders built for all splits",
                    set(loaders) == {"train", "val", "test"}, str(sorted(loaders)))

        batch = next(iter(loaders["train"]))
        ok &= check("batch clip shape",
                    tuple(batch["clip"].shape[1:]) == (8, 3, 112, 112),
                    str(tuple(batch["clip"].shape)))

        if quick:
            return ok

        model = build_model(cfg)
        from fsvnet.train import train
        hist = train(model, loaders, cfg,
                     class_weights=loaders["train"].dataset.class_weights(),
                     run_name="smoke")
        ok &= check("training ran and saved a checkpoint",
                    Path(hist["best_checkpoint"]).exists())

        from fsvnet.evaluate import evaluate_model
        res = evaluate_model(model, loaders, cfg, ("val", "test"))
        ok &= check("evaluation produced a finite test AUC",
                    "test" in res and np.isfinite(res["test"]["auc"]),
                    f"AUC {res.get('test', {}).get('auc')}")

        # ---- 5. evidence + explanation -------------------------------
        print("\n5. Evidence pack and reasoning fallback")
        from fsvnet.evidence import build_evidence
        from fsvnet.vlm_reasoner import EvidenceReasoner, groundedness_check
        b = next(iter(loaders["test"]))
        clip1 = b["clip"][:1].to(cfg.device)
        mask1 = b["mask"][:1].to(cfg.device)
        pack = build_evidence(model, clip1, mask1, "smoke_test.mp4", cfg,
                              reference=None, frames_uint8=None,
                              make_figures=False)
        ok &= check("evidence pack has calibratable descriptors",
                    len(pack.descriptors) >= 5, str(sorted(pack.descriptors)))
        ok &= check("evidence pack warns when uncalibrated",
                    any("reference" in n.lower() for n in pack.notes))

        cfg.vlm.fallback_chain = ["template"]
        expl = EvidenceReasoner(cfg).explain(pack, None)
        ok &= check("reasoning degrades to the template backend",
                    expl.backend == "template" and len(expl.text) > 40)
        g = groundedness_check(expl.text, pack)
        ok &= check("groundedness check runs", "grounded_score" in g,
                    f"score {g.get('grounded_score')}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the training step (still builds and caches)")
    args = ap.parse_args()

    print("=" * 70)
    print("FSV-Net smoke test")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 70)

    ok = True
    ok &= test_tsc_controlled()
    ok &= test_model_shapes()
    ok &= test_splits()
    ok &= test_end_to_end(args.quick)

    print("\n" + "=" * 70)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    for name, status, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name}  {detail}")
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
