#!/usr/bin/env python
"""
finetune_video.py
=================

Robustness fine-tune of the deployed FSV-Net checkpoint (C1 by default) for the
website, then a fresh calibration and abstention band, then a ship decision
that is fixed BEFORE the run.

Why a fine-tune and not a retrain
---------------------------------
The website feeds the model face crops made by a *different* face detector
(the app's MediaPipe/Haar crop with a 30% margin) from *web* video (re-encoded,
resized, sometimes soft). The checkpoint was trained on Celeb-DF-v2's own
pre-extracted crops. So the gap to close is a deployment gap, not a capacity
gap, and the cheapest honest fix is to show the model what deployment looks
like while keeping everything it already learned:

* crop jitter       random rescale (0.85-1.20) and shift (+-8%) of the 256 px
                    crop before resizing to 224 -- a different detector's box
* web degradation   JPEG q 30-95, down/up-scale 0.35-1, Gaussian blur, sensor
                    noise, brightness/contrast/gamma, frame drop/duplication
* class balance     focal loss with inverse-frequency class weights (train is
                    91.6% fake; uploads are not)
* small steps       early EfficientNet blocks frozen; backbone LR 1e-5, rest 1e-4

Model selection uses the mean of clean and *degraded* val AUC, so a model that
gets robust by forgetting Celeb-DF does not win.

Ship rule (fixed before any run; printed in the report)
-------------------------------------------------------
The fine-tuned model replaces the deployed one only if, on the identity-disjoint
TEST split,
  1. degraded-test AUC improves, paired-bootstrap 95% CI excluding 0, and
  2. clean-test AUC does not drop by more than 0.005 (point estimate).
Otherwise the report says so and only the refreshed calibration of the
ORIGINAL checkpoint is packaged. A negative result is a result.

Run (RTX 4060, 8 GB):
    python scripts/finetune_video.py --smoke                  # 2 min wiring check
    python scripts/finetune_video.py                          # full (~3-5 h)
    python scripts/finetune_video.py --calibrate-only         # re-fit calibration of C1 only
    python scripts/finetune_video.py --resume                 # continue after interruption

Outputs: results/finetune_video/<run>/
    report.md, report.json, scores.npz, history.json
    release/  weights.pt + calibration.json (+ handler files if --plugin-src is given)
    release.zip  -> Admin page -> Replace on the FSV-Net model
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import Config, celebdf_frames as cf                      # noqa: E402
from fsvnet.benchmark import paired_bootstrap                         # noqa: E402
from fsvnet.calibrate import pick_calibrator, sigmoid                 # noqa: E402
from fsvnet.dataset_frames import augment, normalise                  # noqa: E402
from fsvnet.losses import FocalLoss                                   # noqa: E402
from fsvnet.model import build_model                                  # noqa: E402
from fsvnet.selective import fit_abstention                           # noqa: E402

MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CKPTS = {"C1": ROOT / "models" / "checkpoints" / "fsvnet_dfe_C1.pt",
         "B0": ROOT / "models" / "checkpoints" / "fsvnet_dfe_B0.pt"}
FALLBACK = {"C1": ROOT / "checkpoints" / "dfe_C1_best.pt",
            "B0": ROOT / "checkpoints" / "dfe_B0_best.pt"}
N_FRAMES, IMG = 16, 224
SHIP_MAX_CLEAN_DROP = 0.005


# ---------------------------------------------------------------------------
# deployment-style augmentation
# ---------------------------------------------------------------------------

def crop_jitter(frames: np.ndarray, rng, scale=(0.85, 1.20), shift=0.08) -> np.ndarray:
    """Same jitter on every frame of the clip (a detector box is temporally smooth)."""
    import cv2
    T, H, W = frames.shape[:3]
    s = float(rng.uniform(*scale))
    dx, dy = (float(rng.uniform(-shift, shift)) * W for _ in range(2))
    side = s * W
    cx, cy = W / 2 + dx, H / 2 + dy
    M = np.array([[W / side, 0, (side / 2 - cx) * W / side],
                  [0, H / side, (side / 2 - cy) * H / side]], dtype=np.float32)
    return np.stack([cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
                     for f in frames])


def web_degrade(frames: np.ndarray, rng, strength: float = 1.0) -> np.ndarray:
    """Blur, noise, photometric change and frame drop/duplicate, each with p~0.3-0.5.
    (JPEG and rescale come from dataset_frames.augment, already part of training.)"""
    import cv2
    out = frames.astype(np.float32)
    if rng.random() < 0.35 * strength:
        sig = float(rng.uniform(0.4, 1.6))
        out = np.stack([cv2.GaussianBlur(f, (0, 0), sig) for f in out])
    if rng.random() < 0.30 * strength:
        out = out + rng.normal(0, float(rng.uniform(1.5, 6.0)), out.shape).astype(np.float32)
    if rng.random() < 0.50 * strength:
        a = float(rng.uniform(0.8, 1.2))
        b = float(rng.uniform(-18, 18))
        g = float(rng.uniform(0.85, 1.15))
        out = np.clip(out * a + b, 0, 255)
        out = 255.0 * (out / 255.0) ** g
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.20 * strength and len(out) > 2:        # variable frame rate
        idx = np.arange(len(out))
        k = int(rng.integers(1, len(out)))
        idx[k] = idx[k - 1]
        out = out[idx]
    return np.ascontiguousarray(out)


def fixed_degrade(frames: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 'typical upload' chain for the degraded val/test split:
    crop jitter, down/up-scale 0.6, JPEG q 50, blur sigma 0.8."""
    import cv2
    rng = np.random.default_rng(10_000 + seed)
    out = crop_jitter(frames, rng, scale=(0.9, 1.15), shift=0.06)
    h, w = out.shape[1:3]
    out = np.stack([cv2.resize(cv2.resize(f, (int(w * 0.6), int(h * 0.6)),
                                          interpolation=cv2.INTER_AREA), (w, h)) for f in out])
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    out = np.stack([cv2.imdecode(cv2.imencode(".jpg", f, enc)[1], 1) for f in out])
    out = np.stack([cv2.GaussianBlur(f, (0, 0), 0.8) for f in out])
    return np.ascontiguousarray(out)


class ClipDS(Dataset):
    def __init__(self, df, split, mode: str, seed: int = 42, limit: int = 0):
        """mode: 'train' | 'clean' | 'degraded'"""
        d = df[df.split == split].reset_index(drop=True)
        if limit:
            import pandas as pd
            d = pd.concat([d[d.label == c].head(max(1, limit // 2)) for c in (0, 1)]
                          ).reset_index(drop=True)
        self.df, self.mode, self.seed = d, mode, seed
        self.epoch = 0

    def __len__(self):
        return len(self.df)

    def class_weights(self) -> torch.Tensor:
        n = np.bincount(self.df.label.values, minlength=2).astype(np.float64)
        return torch.tensor(n.sum() / (2.0 * np.maximum(n, 1)), dtype=torch.float32)

    def __getitem__(self, i):
        import cv2
        r = self.df.iloc[i]
        train = self.mode == "train"
        rng = np.random.default_rng((self.seed + 7919 * self.epoch + i) if train else i)
        frames, _, _ = cf.load_clip(r.video_path, None, n_frames=N_FRAMES, img_size=256,
                                    mode="jitter" if train else "uniform", rng=rng)
        if train:
            if rng.random() < 0.8:
                frames = crop_jitter(frames, rng)
        elif self.mode == "degraded":
            frames = fixed_degrade(frames, i)
        # the app resizes its 256 px crop to 224 with INTER_AREA; do the same
        frames = np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA)
                           for f in frames])
        if train:
            frames = augment(frames, rng, jpeg_quality=(30, 95), downscale=(0.35, 1.0), p=0.5)
            frames = web_degrade(frames, rng)
        return {"clip": normalise(frames), "label": torch.tensor(int(r.label))}


# ---------------------------------------------------------------------------

def load_model(path: Path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict") or blob.get("model_state_dict") or blob
    use_tsc = bool(blob.get("use_tsc", any(k.startswith("tsc.") for k in sd)))
    use_lmf = bool(blob.get("use_lmf", False))
    if use_lmf:
        raise SystemExit("this script fine-tunes landmark-free checkpoints (B0/C1) only: "
                         "uploads carry no 81-point landmarks")
    cfg = Config()
    cfg.data.num_frames, cfg.data.img_size = N_FRAMES, IMG
    cfg.model.use_tsc, cfg.model.use_lmf, cfg.model.pretrained = use_tsc, False, False
    cfg.model.init_from_fswnet = None
    cfg.train.grad_checkpointing = False
    m = build_model(cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return m.to(device), blob, cfg


@torch.no_grad()
def score(model, loader, device, amp: bool):
    model.eval()
    ys, zs = [], []
    for b in loader:
        x = b["clip"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(x, lmf=None)
        logits = logits.float().cpu().numpy()
        zs.append(logits[:, 1] - logits[:, 0])
        ys.append(b["label"].numpy())
    return np.concatenate(ys), np.concatenate(zs)


def auc(y, z):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else float("nan")


def param_groups(model, freeze_blocks: int, lr: float, backbone_lr: float):
    feats = model.spatial.features
    for i, blk in enumerate(feats):
        for p in blk.parameters():
            p.requires_grad = i >= freeze_blocks
    back = [p for p in feats.parameters() if p.requires_grad]
    back_ids = {id(p) for p in feats.parameters()}
    rest = [p for p in model.parameters() if id(p) not in back_ids and p.requires_grad]
    return [{"params": back, "lr": backbone_lr}, {"params": rest, "lr": lr}]


def calibrate(y_val, z_val, alpha: float):
    cal, cv = pick_calibrator(z_val, y_val)
    p_val = cal.transform(z_val)
    pol = fit_abstention(p_val, y_val, alpha=alpha, class_conditional=True)
    prm = cal.params if isinstance(cal.params, dict) else {}
    if cal.name == "platt":
        a, b = prm["a"], prm["b"]
    elif cal.name == "temperature":
        a, b = 1.0 / prm["T"], 0.0
    else:
        a, b = 1.0, 0.0
    return {"method": cal.name, "a": round(a, 4), "b": round(b, 4),
            "band": [round(float(pol.tau_real), 4), round(float(pol.tau_fake), 4)],
            "fit_prior": round(float(y_val.mean()), 4), "deploy_prior": 0.5,
            "alpha": alpha, "cv_nll": {k: round(v, 4) for k, v in cv.items()}}


def selective(cal, y, z):
    p = sigmoid(cal["a"] * z + cal["b"])
    lo, hi = cal["band"]
    ans = (p <= lo) | (p >= hi)
    pred = (p >= hi).astype(int)
    cov = float(ans.mean())
    acc = float((pred[ans] == y[ans]).mean()) if ans.any() else float("nan")
    return cov, acc


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="C1", choices=list(CKPTS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--freeze-blocks", type=int, default=4,
                    help="EfficientNet-B0 feature blocks 0..N-1 stay frozen (of 9)")
    ap.add_argument("--alpha", type=float, default=0.10, help="selective error target")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="clips per split (debug)")
    ap.add_argument("--smoke", action="store_true", help="--limit 16 --epochs 1, CPU ok")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", default=time.strftime("%Y%m%d_%H%M"))
    ap.add_argument("--manifest", default=str(MAN))
    ap.add_argument("--plugin-src", default="",
                    help="folder with handler.py / fsv_*.py / narrate.py to bundle into release.zip "
                         "(e.g. ../mainUI1_xai/backend/plugins_src/fsvnet_video)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.limit, a.epochs, a.workers = a.limit or 16, 1, 0

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = device.type == "cuda"
    out = ROOT / "results" / "finetune_video" / a.run
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | run dir {out}")
    torch.manual_seed(42)
    np.random.seed(42)

    base_path = CKPTS[a.base] if CKPTS[a.base].exists() else FALLBACK[a.base]
    df = cf.load_manifest(a.manifest, ROOT)
    mk = lambda split, mode: DataLoader(ClipDS(df, split, mode, limit=a.limit),
                                        batch_size=a.batch, shuffle=(mode == "train"),
                                        num_workers=a.workers, pin_memory=amp,
                                        drop_last=(mode == "train"))
    evals = {f"{s}_{m}": mk(s, m) for s in ("val", "test") for m in ("clean", "degraded")}

    # ---- baseline scores (the deployed model) -----------------------------
    base, base_blob, cfg = load_model(base_path, device)
    t0 = time.time()
    S = {f"base_{k}": score(base, l, device, amp) for k, l in evals.items()}
    print(f"baseline scored in {time.time() - t0:.0f}s: " +
          ", ".join(f"{k} AUC {auc(*v):.4f}" for k, v in S.items()))
    cal_base = calibrate(*S["base_val_clean"], a.alpha)
    report = {"base_checkpoint": str(base_path), "ship_rule": {
        "degraded_test_auc": "improves with paired-bootstrap 95% CI excluding 0",
        "clean_test_auc_max_drop": SHIP_MAX_CLEAN_DROP}, "calibration_base": cal_base}

    best_path = out / "finetuned_best.pt"
    hist = []
    if not a.calibrate_only:
        model, _, _ = load_model(base_path, device)
        train_ds = ClipDS(df, "train", "train", limit=a.limit)
        train = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                           pin_memory=amp, drop_last=True)
        crit = FocalLoss(train_ds.class_weights().to(device), gamma=2.0, label_smoothing=0.05)
        opt = torch.optim.AdamW(param_groups(model, a.freeze_blocks, a.lr, a.backbone_lr),
                                weight_decay=1e-4)
        steps = max(1, a.epochs * len(train) // a.accum)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[a.backbone_lr, a.lr], total_steps=steps + 1, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        start, best = 0, -1.0
        ck = out / "last.pt"
        if a.resume and ck.exists():
            st = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); start, best, hist = st["epoch"] + 1, st["best"], st["hist"]
            print(f"resumed at epoch {start}")
        for ep in range(start, a.epochs):
            model.train()
            train_ds.epoch = ep
            t0, run_loss, n = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for it, b in enumerate(train):
                x, y = b["clip"].to(device, non_blocking=True), b["label"].to(device)
                with torch.autocast(device.type, enabled=amp):
                    logits, _ = model(x, lmf=None)
                    loss = crit(logits.float(), y) / a.accum
                scaler.scale(loss).backward()
                if (it + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    sched.step()
                run_loss += float(loss) * a.accum; n += 1
                if it % 200 == 0:
                    print(f"  ep {ep} it {it}/{len(train)} loss {run_loss / n:.4f} "
                          f"({(time.time() - t0) / max(it, 1):.2f}s/it)", flush=True)
            vc = auc(*score(model, evals["val_clean"], device, amp))
            vd = auc(*score(model, evals["val_degraded"], device, amp))
            sel = (vc + vd) / 2
            hist.append({"epoch": ep, "loss": run_loss / max(n, 1), "val_auc_clean": vc,
                         "val_auc_degraded": vd, "select": sel, "secs": time.time() - t0})
            print(f"epoch {ep}: loss {run_loss / max(n, 1):.4f} | val AUC clean {vc:.4f} "
                  f"degraded {vd:.4f} | select {sel:.4f}")
            if sel > best:
                best = sel
                torch.save({"state_dict": model.state_dict()}, best_path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best": best, "hist": hist}, ck)
        (out / "history.json").write_text(json.dumps(hist, indent=2))

    # ---- evaluate fine-tuned, decide --------------------------------------
    ship = False
    if best_path.exists() and not a.calibrate_only:
        ft, _, _ = load_model(base_path, device)
        ft.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
        ft.to(device)
        for k, l in evals.items():
            S[f"ft_{k}"] = score(ft, l, device, amp)
        cmp = {}
        for k in ("test_clean", "test_degraded", "val_clean", "val_degraded"):
            y, zb = S[f"base_{k}"]
            _, zf = S[f"ft_{k}"]
            bs = paired_bootstrap(y, zf, zb, n=2000)
            cmp[k] = {"auc_base": auc(y, zb), "auc_ft": auc(y, zf), **bs}
        report["comparison"] = cmp
        d_deg, d_clean = cmp["test_degraded"], cmp["test_clean"]
        lo = d_deg.get("ci_lo", d_deg.get("ci_low", -1))
        ship = (lo > 0) and (d_clean["auc_ft"] - d_clean["auc_base"] >= -SHIP_MAX_CLEAN_DROP)
        report["ship"] = ship
        report["calibration_ft"] = calibrate(*S["ft_val_clean"], a.alpha)

    # ---- package -----------------------------------------------------------
    rel = out / "release"
    rel.mkdir(exist_ok=True)
    use_ft = ship
    cal = report["calibration_ft"] if use_ft else cal_base
    yk = "ft_test_clean" if use_ft else "base_test_clean"
    cov, acc = selective(cal, *S[yk])
    cal.update({"fit_split": f"Celeb-DF-v2 identity-disjoint val (n={len(S['base_val_clean'][0])}), "
                              f"{N_FRAMES} uniform frames at {IMG} px",
                "test_coverage": round(cov, 4), "test_selective_accuracy": round(acc, 4),
                "applies_to_checkpoint": "finetuned (" + a.run + ")" if use_ft else base_path.name,
                "source": "scripts/finetune_video.py"})
    val_auc = auc(*S["ft_val_clean" if use_ft else "base_val_clean"])
    cal_ok = cal["a"] > 0 and val_auc >= 0.6
    if cal_ok:
        (rel / "calibration.json").write_text(json.dumps(cal, indent=2))
    else:
        print(f"NOT packaging calibration: val AUC {val_auc:.3f}, slope {cal['a']}")
    if use_ft:
        sd = torch.load(best_path, map_location="cpu")["state_dict"]
        torch.save({"state_dict": sd, "config": base_blob.get("config"),
                    "condition": f"{a.base}-ft-{a.run}", "use_tsc": cfg.model.use_tsc,
                    "use_lmf": False, "finetuned_from": base_path.name,
                    "monitor": "mean(val_auc_clean, val_auc_degraded)"}, rel / "weights.pt")
    else:
        shutil.copy(base_path, rel / "weights.pt")
    if a.plugin_src:
        src = Path(a.plugin_src)
        for f in list(src.glob("*.py")) + [src / "manifest.json"]:
            if f.exists():
                shutil.copy(f, rel / f.name)
    shutil.make_archive(str(out / "release"), "zip", rel)

    np.savez(out / "scores.npz", **{f"{k}_y": v[0] for k, v in S.items()},
             **{f"{k}_z": v[1] for k, v in S.items()})
    report["packaged"] = "finetuned" if use_ft else "original checkpoint + refreshed calibration"
    report["release_test_selective"] = {"coverage": cov, "accuracy": acc}
    report["history"] = hist
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    lines = [f"# Video fine-tune report ({a.run})", "",
             f"Base checkpoint: `{base_path.name}`  |  device {device}", "",
             "Ship rule (fixed before the run): degraded-test AUC must improve with the paired-"
             f"bootstrap 95% CI excluding 0, and clean-test AUC may not drop by more than "
             f"{SHIP_MAX_CLEAN_DROP}.", ""]
    if "comparison" in report:
        lines += ["| split | base AUC | fine-tuned AUC | diff [95% CI] | p |", "|---|---|---|---|---|"]
        for k, c in report["comparison"].items():
            lines.append(f"| {k} | {c['auc_base']:.4f} | {c['auc_ft']:.4f} | "
                         f"{c.get('diff', c['auc_ft'] - c['auc_base']):+.4f} "
                         f"[{c.get('ci_lo', c.get('ci_low', float('nan'))):+.4f}, "
                         f"{c.get('ci_hi', c.get('ci_high', float('nan'))):+.4f}] | "
                         f"{c.get('p', c.get('p_value', float('nan'))):.3f} |")
        lines += ["", f"**Decision: {'SHIP the fine-tuned model' if ship else 'KEEP the original model'}**", ""]
    lines += [f"Packaged calibration: {cal['method']} a={cal['a']} b={cal['b']}, band "
              f"{cal['band']} (val prior {cal['fit_prior']}); on test it answers "
              f"{cov * 100:.1f}% of clips at {acc * 100:.1f}% selective accuracy.", "",
              f"Install: Admin page -> FSV-Net -> Replace -> `{out / 'release.zip'}`"]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
