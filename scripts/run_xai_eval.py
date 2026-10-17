"""
Explanation benchmark on the identity-disjoint Celeb-DF v2 split.

    python scripts/test_xai_eval.py                      # 27 known-answer checks first
    python scripts/run_xai_eval.py --n 8 --n-stability 2 --arb-limit 32   # smoke
    python scripts/run_xai_eval.py                       # full: C3 spatial study + 4-model panel
    python scripts/run_xai_eval.py --stages spatial      # one stage; REPORT.md still
    python scripts/run_xai_eval.py --stages arbitration  #   includes the other if on disk
    python scripts/run_xai_eval.py --resummarise         # rebuild tables/figures, no GPU
    python scripts/run_xai_eval.py --cpu-fallback        # n=32, no KernelSHAP
    python scripts/run_xai_eval.py --demo                # plumbing check, random weights

Reads models/checkpoints/fsvnet_dfe_<ID>.pt, data/manifests/celebdf_v2_identity.csv
and data/processed/lmf_cache_16f.npz, exactly as scripts/eval_conditions.py does.
If results/scores_<ID>.npz exists it is used to stratify the sample; otherwise
the test split is scored once first.

Stage 1 — spatial (one model, default C3, stratified test subsample)
    attributions : Grad-CAM, IG (blur / mean-face baselines, 32 steps),
                   patch occlusion, KernelSHAP (512 coalitions)
    controls     : random x3, centre prior, blur-everything
    metrics      : deletion / insertion AUC, comprehensiveness, sufficiency,
                   flip fraction — under blur, mean-face and roll fills
    statistics   : paired bootstrap vs random and vs centre (IPW-weighted),
                   Borda ranking, fill agreement
    also         : stability (sigma=0.01 noise, resampled frames, KernelSHAP
                   reseed); model-randomisation sanity check; spectral /
                   temporal / region counterfactuals; temporal faithfulness
                   (occlusion vs attention vs random); does faithfulness
                   predict correctness beyond confidence; modality gate

Stage 2 — arbitration (every available checkpoint, full val + test)
    per-clip exact branch Shapley; reliability matrix (model x channel);
    Platt-calibrated single models vs two stacks, all fitted on val only;
    compared with (a) the best single model chosen by 5-fold CV NLL on val and
    (b) the pre-registered B0 baseline; paired bootstrap on test.

Outputs: results/xai/*.csv, *.json, figures/*.png, REPORT.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import xai_eval as X  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
CKPT_DIR = ROOT / "models" / "checkpoints"
MODEL_IDS = ["B0", "C1", "C2", "C3"]
FILLS = ["blur", "mean", "roll"]
PRIMARY_FILLS = ["blur", "mean"]          # the winner must beat random under both
METHODS = ["gradcam", "ig_blur", "ig_mean", "occlusion", "kshap"]
BORDA_METRICS = {"del_auc": "lower", "ins_auc": "higher", "comp_mean": "higher",
                 "suff_abs": "lower", "flip_frac": "lower"}
# Written in claude/cfx_implementation.md §4 BEFORE any run. Checked, never edited.
PREREGISTERED_ROUTING = "B0"
PREREGISTERED_BASELINE = "B0"
# A stack must beat a single model by at least this much, with the 95% CI also
# excluding zero, before it is called better. Fixed before any real run: a CI
# can exclude zero on a difference far too small to matter (the random-weight
# demo resolved a Brier gap of 2e-8), and that is not a reason to ship a stack.
MIN_DAUC, MIN_DBRIER = 0.005, 0.002


# ==========================================================================
# data sources
# ==========================================================================

class RealSource:
    def __init__(self, split, T, S, cache_path, seed=42):
        from fsvnet import celebdf_frames as cf
        from fsvnet.dataset_frames import CelebDFFramesDataset, LMFCache, normalise
        self.cf, self.normalise = cf, normalise
        df = cf.load_manifest(MAN, ROOT)
        self.ds = CelebDFFramesDataset(df, split, T, S, augment_train=False,
                                       lmf_cache=LMFCache(cache_path, T), seed=seed)
        self.T, self.S = T, S
        self.labels = self.ds.df.label.values.astype(int)

    def __len__(self):
        return len(self.ds)

    def item(self, i):
        return self.ds[i]

    def resample(self, i):
        """Same clip, a different set of frames — for the stability test."""
        r = self.ds.df.iloc[i]
        fr, _, _ = self.cf.load_clip(r.video_path, None, n_frames=self.T, img_size=self.S,
                                     mode="jitter", rng=np.random.default_rng(10_000 + i))
        return self.normalise(fr)


class DemoSource:
    """Synthetic clips with a planted bright patch on fakes. Plumbing only."""
    def __init__(self, n, T, S, seed):
        self.T, self.S, self.n, self.seed = T, S, n, seed
        self.labels = np.random.default_rng(seed).integers(0, 2, n)

    def __len__(self):
        return self.n

    def _clip(self, i, salt=0):
        g = torch.Generator().manual_seed(self.seed * 7919 + i * 31 + salt)
        x = torch.randn(self.T, 3, self.S, self.S, generator=g) * 0.5
        if self.labels[i]:
            x[:, :, self.S // 4:self.S // 2, self.S // 2:3 * self.S // 4] += 1.5
        return x

    def item(self, i):
        g = torch.Generator().manual_seed(i)
        return {"clip": self._clip(i), "label": torch.tensor(int(self.labels[i])),
                "lmf_energy": torch.randn(self.T - 1, 3, generator=g),
                "lmf_desc": torch.randn(7, generator=g), "lmf_ok": torch.tensor(True),
                "clip_name": f"demo_{self.seed}_{i:04d}"}

    def resample(self, i):
        return self._clip(i, salt=1)


# ==========================================================================
# models
# ==========================================================================

def load_model(cid, T, S, device, demo=False):
    from fsvnet import Config
    from fsvnet.model import build_model
    c = Config()
    c.data.num_frames, c.data.img_size = T, S
    c.model.pretrained, c.model.init_from_fswnet = False, None
    if demo:
        c.model.use_tsc, c.model.use_lmf = {"B0": (False, False), "C1": (True, False),
                                            "C2": (False, True), "C3": (True, True)}[cid]
        torch.manual_seed(MODEL_IDS.index(cid))
        m = build_model(c)
        with torch.no_grad():                 # a random head is near-constant
            m.classifier[-1].weight.mul_(20)
        return m.to(device).eval(), {"demo": True}
    ck = CKPT_DIR / f"fsvnet_dfe_{cid}.pt"
    if not ck.exists():
        return None, None
    blob = torch.load(ck, map_location=device, weights_only=False)
    c.model.use_tsc, c.model.use_lmf = blob["use_tsc"], blob["use_lmf"]
    m = build_model(c)
    m.load_state_dict(blob["state_dict"])
    return m.to(device).eval(), {"checkpoint": ck.name, "epoch": blob.get("epoch")}


def lmf_of(model, item, device):
    if not model.use.get("lmf"):
        return None
    return {"energy": item["lmf_energy"][None].to(device),
            "descriptors": item["lmf_desc"][None].to(device)}


def batch_lmf(model, items, device):
    if not model.use.get("lmf"):
        return None
    return {"energy": torch.stack([it["lmf_energy"] for it in items]).to(device),
            "descriptors": torch.stack([it["lmf_desc"] for it in items]).to(device)}


def score_split(model, src, device, batch=4):
    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(src), batch):
            items = [src.item(j) for j in range(i, min(i + batch, len(src)))]
            lo, _ = model(torch.stack([it["clip"] for it in items]).to(device), None,
                          lmf=batch_lmf(model, items, device))
            ps += torch.softmax(lo.float(), 1)[:, 1].cpu().tolist()
            ys += [int(it["label"]) for it in items]
    return np.array(ys), np.array(ps)


# ==========================================================================
# sampling
# ==========================================================================

def stratified_sample(y, p, n, seed=42):
    """
    Over-sample errors (up to a third of the sample) so "does faithfulness
    predict correctness" has power, and return inverse-probability weights so
    population means stay unbiased.
    """
    rng = np.random.default_rng(seed)
    correct = ((p >= 0.5).astype(int) == y)
    strata = {(lab, ok): np.where((y == lab) & (correct == ok))[0]
              for lab in (0, 1) for ok in (True, False)}
    n_err_avail = sum(len(strata[(l, False)]) for l in (0, 1))
    n_err = min(n_err_avail, n // 3)
    alloc = {}
    for l in (0, 1):
        share = len(strata[(l, False)]) / max(n_err_avail, 1)
        alloc[(l, False)] = min(len(strata[(l, False)]), int(round(n_err * share)))
    rest = n - sum(alloc.values())
    for l in (0, 1):
        alloc[(l, True)] = min(len(strata[(l, True)]), rest // 2)
    short = n - sum(alloc.values())                    # top up if a stratum ran dry
    for key in sorted(strata, key=lambda k: -len(strata[k])):
        extra = min(short, len(strata[key]) - alloc[key]); alloc[key] += extra; short -= extra
    idx, w = [], []
    for key, k in alloc.items():
        if k:
            idx += list(rng.choice(strata[key], k, replace=False))
            w += [len(strata[key]) / k] * k
    order = rng.permutation(len(idx))
    comp = {f"{'fake' if l else 'real'}_{'correct' if ok else 'error'}":
            {"population": int(len(strata[(l, ok)])), "sampled": int(alloc[(l, ok)])}
            for (l, ok) in strata}
    return np.array(idx)[order], np.array(w)[order], comp


def mean_face(src, device, n=100):
    """ImageNet-normalised mean face over real clips (middle frame), from VAL."""
    real = np.where(src.labels == 0)[0][:n]
    acc = None
    for i in real:
        f = src.item(int(i))["clip"][src.T // 2]
        acc = f.clone() if acc is None else acc + f
    return (acc / max(len(real), 1))[None, None].to(device)


def to_u8(frame):
    x = frame.numpy().transpose(1, 2, 0) * X.IMAGENET_STD + X.IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ==========================================================================
# stage 1 — spatial faithfulness and friends
# ==========================================================================

def explain_one(model, clip, lmf, masks, fills, args, seed):
    """All unit-level attributions for one clip."""
    sc = X.ClipScorer(model, clip, lmf, batch=args.batch, amp=args.amp)
    out, extra = {}, {}
    t0 = time.time()
    g = X.grad_cam_units(model, clip, lmf, sc.sign, masks)
    if g is not None:
        out["gradcam"] = g
    for b in ("blur", "mean"):
        out[f"ig_{b}"], extra[f"ig_{b}_resid"] = X.integrated_gradients_units(
            model, clip, lmf, sc.sign, fills[b], masks, steps=args.ig_steps, batch=args.ig_batch)
    out["occlusion"] = X.occlusion_units(sc, fills["blur"], masks)
    if args.kshap:
        out["kshap"] = X.kernel_shap_units(sc, fills["blur"], masks, args.kshap, seed=seed)
    extra["attr_seconds"] = time.time() - t0
    return sc, out, extra


def stage_spatial(args, device, out_dir):
    T, S, G = args.frames, args.size, args.grid
    print(f"\n=== stage 1: spatial faithfulness — model {args.model} ===")
    model, meta = load_model(args.model, T, S, device, demo=args.demo)
    if model is None:
        print(f"  no checkpoint for {args.model}; stage skipped"); return
    src = DemoSource(40, T, S, 1) if args.demo else RealSource("test", T, S, args.lmf_cache)
    val_src = DemoSource(40, T, S, 2) if args.demo else RealSource("val", T, S, args.lmf_cache)

    npz = ROOT / "results" / f"scores_{args.model}.npz"
    if not args.demo and npz.exists() and len(np.load(npz)["p_test"]) == len(src):
        z = np.load(npz); y_all, p_all = z["y_test"].astype(int), z["p_test"]
        print(f"  stratifying on {npz.name}")
    else:
        print(f"  scoring all {len(src)} test clips for stratification ...")
        y_all, p_all = score_split(model, src, device, args.batch)
        if not args.demo:
            np.savez(out_dir / f"test_scores_{args.model}.npz", y_test=y_all, p_test=p_all)
    idx, w, comp = stratified_sample(y_all, p_all, min(args.n, len(src)), seed=42)
    print("  sample:", json.dumps(comp))

    masks = X.unit_masks(S, S, G)
    mface = mean_face(val_src, device)
    rng = np.random.default_rng(0)
    rand_models = {lvl: X.randomized_copy(model, lvl) for lvl in ("head", "head+block")}
    n_stab = min(args.n_stability, len(idx))
    per_clip, rows, temporal_rows, stab_rows, sanity_rows = [], [], [], [], []
    curve_store, examples = {}, {}
    t_start = time.time()

    for j, (i, wt) in enumerate(zip(idx, w)):
        item = src.item(int(i))
        clip = item["clip"][None].to(device)
        lmf = lmf_of(model, item, device)
        y = int(item["label"])
        fills = {f: X.make_fill(f, clip, mface) for f in FILLS}
        sc, units, extra = explain_one(model, clip, lmf, masks, fills, args, seed=j)
        units["centre"] = X.centre_prior_units(G)

        with torch.no_grad():
            _, info = model(clip, None, lmf=lmf, return_evidence=True)
        sh = X.branch_shapley(model, info["embeddings"])
        phi = np.abs(sh["phi"][0]); share = phi / max(phi.sum(), 1e-12)
        shares = dict(zip(sh["names"], share))
        fw = info.get("frame_weights", {})
        attn = (np.mean([v[0].float().cpu().numpy() for v in fw.values()], 0)
                if fw else np.ones(T) / T)
        p_fake = float(X.sigmoid(sc.margin0))

        rec = {"clip_id": str(item["clip_name"]), "label": y, "p_fake": p_fake,
               "p_fake_eval": float(p_all[i]), "correct": int((p_fake >= 0.5) == bool(y)),
               "weight": float(wt), "p_target": sc.p0, "lmf_ok": bool(item.get("lmf_ok", True)),
               "spatial_share": float(shares.get("spatial", np.nan)),
               "pixel_reachable_share": float(1 - shares.get("lmf", 0.0)),
               "shapley_resid": float(sh["efficiency_residual"][0]),
               "shapley_vs_forward": abs(float(sh["v_full"][0]) - sc.margin0),
               **{f"share_{k}": float(v) for k, v in shares.items()}, **extra}

        # faithfulness curves: every method and control, every fill
        rand_orders = [rng.permutation(len(masks)) for _ in range(3)]
        for f in FILLS:
            for mname, u in units.items():
                c = X.curves(sc, fills[f], masks, X.ranking(u, rng))
                rows.append({"clip_id": rec["clip_id"], "method": mname, "fill": f,
                             **X.summarise_curves(c, sc.p0)})
                curve_store.setdefault((mname, f), []).append(np.stack([c["deletion"], c["insertion"]]))
            rc = [X.curves(sc, fills[f], masks, o) for o in rand_orders]
            rs = [X.summarise_curves(c, sc.p0) for c in rc]
            rows.append({"clip_id": rec["clip_id"], "method": "random", "fill": f,
                         **{k: _nanmean([r[k] for r in rs]) for k in rs[0]}})
            curve_store.setdefault(("random", f), []).append(
                np.mean([np.stack([c["deletion"], c["insertion"]]) for c in rc], 0))
            rec[f"all_filled_p_{f}"] = float(rc[0]["insertion"][0])   # nothing kept
        rec["blur_everything_drop"] = sc.p0 - rec["all_filled_p_blur"]

        # temporal faithfulness and counterfactuals
        tsc = X.temporal_scores(sc)
        ks = [k for k in (1, 2, 4, 8) if k < T]
        occ_c = X.temporal_curve(sc, X.ranking(tsc), ks)
        att_c = X.temporal_curve(sc, np.argsort(-attn), ks)
        rnd_c = X.temporal_curve(sc, rng.permutation(T), ks)
        temporal_rows.append({"clip_id": rec["clip_id"], "abs_margin": abs(sc.margin0),
                              **{f"occ@{k}": v for k, v in zip(ks, occ_c)},
                              **{f"att@{k}": v for k, v in zip(ks, att_c)},
                              **{f"rnd@{k}": v for k, v in zip(ks, rnd_c)},
                              "rho_att_occ": X.spearman(attn, tsc),
                              "max_single_frame": float(np.max(tsc))})
        rec["temporal_cf_frames"] = X.temporal_counterfactual(sc, X.ranking(tsc))
        spec = X.spectral_counterfactual(sc)
        rec["spectral_cf_nyq"], rec["spectral_cf_monotone"] = spec["cutoff_nyq"], spec["monotone"]

        # stability + sanity on the first n_stab clips
        if j < n_stab:
            noisy = X.add_pixel_noise(clip, 0.01, seed=j)
            _, u_noise, _ = explain_one(model, noisy, lmf, masks,
                                        {f: X.make_fill(f, noisy, mface) for f in FILLS}, args, j)
            res = src.resample(int(i))[None].to(device)
            _, u_res, _ = explain_one(model, res, lmf, masks,
                                      {f: X.make_fill(f, res, mface) for f in FILLS}, args, j)
            for mname in METHODS:
                if mname in units:
                    stab_rows.append({"clip_id": rec["clip_id"], "method": mname,
                                      "rho_noise": X.spearman(units[mname], u_noise[mname]),
                                      "jac_noise": X.topk_jaccard(units[mname], u_noise[mname]),
                                      "rho_frames": X.spearman(units[mname], u_res[mname]),
                                      "jac_frames": X.topk_jaccard(units[mname], u_res[mname])})
            if args.kshap:
                k2 = X.kernel_shap_units(sc, fills["blur"], masks, args.kshap, seed=j + 991)
                stab_rows.append({"clip_id": rec["clip_id"], "method": "kshap_reseed",
                                  "rho_noise": X.spearman(units["kshap"], k2),
                                  "jac_noise": X.topk_jaccard(units["kshap"], k2)})
            for lvl, rm in rand_models.items():
                g = X.grad_cam_units(rm, clip, lmf, sc.sign, masks)
                ig, _ = X.integrated_gradients_units(rm, clip, lmf, sc.sign, fills["blur"], masks,
                                                     steps=args.ig_steps, batch=args.ig_batch)
                sanity_rows.append({"clip_id": rec["clip_id"], "level": lvl,
                                    "rho_gradcam": (X.spearman(units["gradcam"], g)
                                                    if g is not None and "gradcam" in units else np.nan),
                                    "rho_ig_blur": X.spearman(units["ig_blur"], ig)})

        rec["forwards"] = sc.n_forward
        per_clip.append(rec)
        examples[rec["clip_id"]] = {"units": units, "frame": to_u8(item["clip"][T // 2])}
        el = time.time() - t_start
        print(f"  [{j + 1:>3}/{len(idx)}] {rec['clip_id'][:30]:<30} y={y} p={p_fake:.3f} "
              f"fwd={sc.n_forward:>4}  {el / (j + 1):.1f}s/clip  "
              f"eta {el / (j + 1) * (len(idx) - j - 1) / 60:.0f} min", flush=True)
        if not args.demo and abs(p_fake - p_all[i]) > 1e-3:
            print(f"      WARNING: p here {p_fake:.4f} vs stratification {p_all[i]:.4f} — "
                  "loader order or preprocessing differs from eval_conditions.py")
        if (j + 1) % 25 == 0:                          # checkpoint long runs
            _dump_raw(out_dir, per_clip, rows, temporal_rows, stab_rows, sanity_rows,
                      examples, comp, meta, curve_store, args, partial=True)

    _dump_raw(out_dir, per_clip, rows, temporal_rows, stab_rows, sanity_rows,
              examples, comp, meta, curve_store, args, partial=False)
    finish_spatial(out_dir)


def _dump_raw(out_dir, per_clip, rows, temporal, stab, sanity, examples, comp, meta,
              curves, args, partial):
    with open(out_dir / "_stage1_raw.pkl", "wb") as fh:
        pickle.dump({"per_clip": per_clip, "rows": rows, "temporal": temporal, "stab": stab,
                     "sanity": sanity, "examples": examples, "comp": comp, "meta": meta,
                     "curves": curves, "args": vars(args), "partial": partial,
                     "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh)


def finish_spatial(out_dir):
    """Tables, figures and summary from _stage1_raw.pkl. CPU only, seconds."""
    with open(out_dir / "_stage1_raw.pkl", "rb") as fh:
        R = pickle.load(fh)
    pc, fr = pd.DataFrame(R["per_clip"]), pd.DataFrame(R["rows"])
    tp, st, sa = pd.DataFrame(R["temporal"]), pd.DataFrame(R["stab"]), pd.DataFrame(R["sanity"])
    pc.to_csv(out_dir / "spatial_per_clip.csv", index=False)
    fr.to_csv(out_dir / "spatial_faithfulness_per_clip.csv", index=False)
    tp.to_csv(out_dir / "temporal_per_clip.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", fractions=X.CURVE_FRACTIONS,
                        **{f"{m}__{f}": np.stack(v) for (m, f), v in R["curves"].items()})

    wmap = dict(zip(pc["clip_id"], pc["weight"]))
    fills = [f for f in FILLS if f in set(fr["fill"])]
    methods = [m for m in METHODS + ["centre", "random"] if m in set(fr["method"])]
    table, borda_by_fill = [], {}
    for f in fills:
        sub = fr[fr["fill"] == f].set_index(["clip_id", "method"])
        rnd = sub.xs("random", level="method")
        cen = sub.xs("centre", level="method")
        agg = {}
        for m in methods:
            s = sub.xs(m, level="method")
            wts = s.index.map(wmap).values.astype(float)
            r, c = rnd.loc[s.index], cen.loc[s.index]
            agg[m] = {k: X.wmean(s[k], wts) for k in BORDA_METRICS}
            d_del = X.paired_diff(r["del_auc"], s["del_auc"], wts)        # > 0: beats random
            d_ins = X.paired_diff(s["ins_auc"], r["ins_auc"], wts)
            d_comp = X.paired_diff(s["comp_mean"], r["comp_mean"], wts)
            d_cen = X.paired_diff(c["del_auc"], s["del_auc"], wts)
            table.append({"method": m, "fill": f,
                          **{k: round(v, 4) for k, v in agg[m].items()},
                          "comp_mean_signed": round(X.wmean(s["comp_mean"], wts), 4),
                          "suff_mean_signed": round(X.wmean(s["suff_mean"], wts), 4),
                          "d_del_vs_random": round(d_del["mean"], 4),
                          "d_del_CI": f"[{d_del['lo']:+.4f}, {d_del['hi']:+.4f}]",
                          "p_del": round(d_del["p"], 4),
                          "d_ins_vs_random": round(d_ins["mean"], 4), "p_ins": round(d_ins["p"], 4),
                          "d_comp_vs_random": round(d_comp["mean"], 4), "p_comp": round(d_comp["p"], 4),
                          "d_del_vs_centre": round(d_cen["mean"], 4), "p_centre": round(d_cen["p"], 4),
                          "beats_random": bool(m != "random" and d_del["lo"] > 0)})
        borda_by_fill[f] = X.borda({m: agg[m] for m in methods}, BORDA_METRICS)
    tab = pd.DataFrame(table)
    tab["borda"] = [borda_by_fill[r.fill][r.method] for r in tab.itertuples()]
    tab.to_csv(out_dir / "spatial_faithfulness.csv", index=False)

    from scipy.stats import kendalltau
    real = [m for m in methods if m in METHODS]

    def order(f):
        return [tab[(tab["fill"] == f) & (tab["method"] == m)]["del_auc"].item() for m in real]
    taus = {f"{a}~{b}": (float(kendalltau(order(a), order(b)).statistic) if len(real) > 2 else None)
            for a, b in (("blur", "mean"), ("blur", "roll"), ("mean", "roll"))
            if a in fills and b in fills}
    prim = [f for f in PRIMARY_FILLS if f in fills]
    beats = {m: {f: bool(tab[(tab["method"] == m) & (tab["fill"] == f)]["beats_random"].item())
                 for f in fills} for m in real}
    total = {m: float(sum(borda_by_fill[f][m] for f in prim)) for m in real}
    eligible = [m for m in real if all(beats[m][f] for f in prim)]
    winner = max(eligible, key=lambda m: total[m]) if eligible else None

    fvc = {}
    if winner:
        g = [fr[fr["fill"] == f].pivot(index="clip_id", columns="method", values="del_auc")
             .pipe(lambda s: s["random"] - s[winner]) for f in prim]
        F = pd.concat(g, axis=1).mean(axis=1).reindex(pc["clip_id"]).values
        fvc = faith_vs_correct(F, pc["correct"].values, np.abs(pc["p_fake"].values - 0.5))
        fvc["per_stratum"] = {}
        for lab in (0, 1):
            for ok in (1, 0):
                msk = (pc["label"].values == lab) & (pc["correct"].values == ok)
                if msk.sum() >= 3:
                    fvc["per_stratum"][f"{'fake' if lab else 'real'}_{'correct' if ok else 'error'}"] = \
                        {"n": int(msk.sum()), "mean_gain_over_random": float(np.nanmean(F[msk]))}

    stab = (st.groupby("method")[[c for c in ("rho_noise", "jac_noise", "rho_frames", "jac_frames")
                                  if c in st]].mean().round(3)) if len(st) else pd.DataFrame()
    stab.to_csv(out_dir / "stability.csv")
    san = (sa.groupby("level")[["rho_gradcam", "rho_ig_blur"]].agg(["mean", "std"]).round(3)
           if len(sa) else pd.DataFrame())
    san.to_csv(out_dir / "sanity.csv")

    ks = sorted({int(c.split("@")[1]) for c in tp.columns if c.startswith("occ@")})
    temporal = {"k_frames_frozen": ks,
                **{name: [round(float(tp[f"{p}@{k}"].mean()), 4) for k in ks]
                   for name, p in (("occlusion", "occ"), ("attention", "att"), ("random", "rnd"))},
                "rho_attention_occlusion_mean": float(tp["rho_att_occ"].mean()),
                "median_max_single_frame": float(tp["max_single_frame"].median()),
                "mean_abs_margin": float(tp["abs_margin"].mean())}
    pf = pc["p_fake"] >= .5
    cf = {"spectral_flip_rate": float(pc["spectral_cf_nyq"].notna().mean()),
          "spectral_median_cutoff_nyq_pred_fake": _med(pc.loc[pf, "spectral_cf_nyq"]),
          "spectral_median_cutoff_nyq_pred_real": _med(pc.loc[~pf, "spectral_cf_nyq"]),
          "spectral_monotone_rate": float(pc["spectral_cf_monotone"].mean()),
          "temporal_flip_rate": float(pc["temporal_cf_frames"].notna().mean()),
          "temporal_median_frames_to_flip": _med(pc["temporal_cf_frames"])}
    if winner:
        wf = fr[fr["method"] == winner]
        cf["region_flip_rate_winner"] = {f: float(wf[wf["fill"] == f]["flip_frac"].notna().mean()) for f in fills}
        cf["region_median_flip_frac_winner"] = {f: _med(wf[wf["fill"] == f]["flip_frac"]) for f in fills}
    controls = {"blur_everything_drop_mean": X.wmean(pc["blur_everything_drop"], pc["weight"]),
                **{f"p_target_all_filled_{f}": X.wmean(pc[f"all_filled_p_{f}"], pc["weight"])
                   for f in fills},
                "p_target_unperturbed": X.wmean(pc["p_target"], pc["weight"])}
    gate = {"mean_spatial_share": float(pc["spatial_share"].mean()),
            "frac_clips_spatial_share_ge_0.5": float((pc["spatial_share"] >= 0.5).mean()),
            "mean_pixel_reachable_share": float(pc["pixel_reachable_share"].mean()),
            "max_shapley_efficiency_residual": float(pc["shapley_resid"].max()),
            "max_shapley_vs_forward_gap": float(pc["shapley_vs_forward"].max()),
            "clips_missing_lmf": int((~pc["lmf_ok"].astype(bool)).sum())}
    ex = R["examples"]
    degenerate = {m: float(np.mean([X.is_degenerate(e["units"][m]) for e in ex.values()
                                    if m in e["units"]])) for m in real}
    summary = {"model": R["meta"], "n": int(len(pc)), "degenerate_map_rate": degenerate, "partial": R["partial"],
               "run_utc": R["run_utc"], "args": R["args"], "sample": R["comp"],
               "winner": winner, "beats_random": beats, "borda_total_primary": total,
               "fill_rank_agreement_kendall_tau": taus, "controls": controls,
               "modality_gate": gate, "faithfulness_vs_correctness": fvc,
               "temporal": temporal, "counterfactuals": cf,
               "ig_completeness_residual_median": {k: float(pc[k].median())
                                                   for k in ("ig_blur_resid", "ig_mean_resid") if k in pc},
               "attr_seconds_per_clip": float(pc["attr_seconds"].mean()),
               "forwards_per_clip": float(pc["forwards"].mean())}
    with open(out_dir / "spatial_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_js)
    figures_spatial(tab, out_dir, R["examples"], pc, tp, fills, R["args"]["grid"])
    print(f"  stage 1 summarised: winner={winner}")


def faith_vs_correct(F, correct, conf, n=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    ok = np.isfinite(F); F, correct, conf = F[ok], correct[ok], conf[ok]
    n_err = int((correct == 0).sum())
    if len(set(correct)) < 2 or n_err < 5 or len(F) < 20:
        return {"n": int(len(F)), "n_errors": n_err,
                "note": "too few errors to test (needs >= 5 errors and >= 20 clips)"}
    rng = np.random.default_rng(0)
    rb, ab = [], []
    for _ in range(n):
        i = rng.integers(0, len(F), len(F))
        if len(set(correct[i])) == 2 and np.std(F[i]) > 0:
            rb.append(np.corrcoef(F[i], correct[i])[0, 1]); ab.append(roc_auc_score(correct[i], F[i]))
    Z = np.c_[conf, F]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    oof_c, oof_cf = np.zeros(len(F)), np.zeros(len(F))
    folds = min(5, n_err)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(Z, correct):
        oof_c[te] = LogisticRegression().fit(Z[tr, :1], correct[tr]).predict_proba(Z[te, :1])[:, 1]
        oof_cf[te] = LogisticRegression().fit(Z[tr], correct[tr]).predict_proba(Z[te])[:, 1]
    d = []
    for _ in range(n):
        i = rng.integers(0, len(F), len(F))
        if len(set(correct[i])) == 2:
            d.append(roc_auc_score(correct[i], oof_cf[i]) - roc_auc_score(correct[i], oof_c[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": int(len(F)), "n_errors": n_err,
            "note": "on the error-oversampled sample; AUROCs are unweighted",
            "point_biserial_r": float(np.corrcoef(F, correct)[0, 1]),
            "r_CI": [float(x) for x in np.percentile(rb, [2.5, 97.5])],
            "auroc_faithfulness": float(roc_auc_score(correct, F)),
            "auroc_faithfulness_CI": [float(x) for x in np.percentile(ab, [2.5, 97.5])],
            "auroc_confidence": float(roc_auc_score(correct, conf)),
            "cv_auc_confidence_only": float(roc_auc_score(correct, oof_c)),
            "cv_auc_confidence_plus_faithfulness": float(roc_auc_score(correct, oof_cf)),
            "delta_auc_added_by_faithfulness": float(np.mean(d)), "delta_CI": [float(lo), float(hi)],
            "verdict": ("faithfulness adds information beyond confidence" if lo > 0 else
                        "no evidence that faithfulness adds to confidence")}


# ==========================================================================
# stage 2 — reliability matrix and arbitration
# ==========================================================================

def cv_platt_nll(m, y, folds=5):
    """Out-of-fold NLL of a Platt calibrator on val — picks the best single model
    on CALIBRATED quality, so an inverted-but-informative model is not discarded."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(m, y):
        oof[te] = LogisticRegression(C=1e6).fit(m[tr, None], y[tr]).predict_proba(m[te, None])[:, 1]
    return float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))


def stage_arbitration(args, device, out_dir):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    T, S = args.frames, args.size
    print("\n=== stage 2: reliability matrix + arbitration ===")
    srcs = ({"val": DemoSource(args.arb_limit or 80, T, S, 3),
             "test": DemoSource(args.arb_limit or 80, T, S, 4)} if args.demo else
            {"val": RealSource("val", T, S, args.lmf_cache),
             "test": RealSource("test", T, S, args.lmf_cache)})
    panel = {}
    for cid in MODEL_IDS:
        model, meta = load_model(cid, T, S, device, demo=args.demo)
        if model is None:
            print(f"  {cid}: no checkpoint — left out of the panel"); continue
        rec = {"names": list(model.branch_names), "meta": meta}
        for split, src in srcs.items():
            n = len(src) if not args.arb_limit else min(args.arb_limit, len(src))
            y, mg, phi, phin, resid = [], [], [], [], []
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, n, args.batch):
                    its = [src.item(j) for j in range(i, min(i + args.batch, n))]
                    clip = torch.stack([it["clip"] for it in its]).to(device)
                    lmf = batch_lmf(model, its, device)
                    lo, info = model(clip, None, lmf=lmf, return_evidence=True)
                    sh = X.branch_shapley(model, info["embeddings"])
                    y += [int(it["label"]) for it in its]
                    mg += (lo[:, 1] - lo[:, 0]).float().cpu().tolist()
                    phi.append(sh["phi"]); resid.append(sh["efficiency_residual"])
                    if split == "val" and i < args.arb_noise_n:
                        _, inf2 = model(X.add_pixel_noise(clip, 0.01, seed=i), None, lmf=lmf,
                                        return_evidence=True)
                        phin.append(X.branch_shapley(model, inf2["embeddings"])["phi"])
            rec[split] = {"y": np.array(y), "margin": np.array(mg), "phi": np.concatenate(phi),
                          "resid": np.concatenate(resid)}
            if phin:
                rec[split]["phi_noisy"] = np.concatenate(phin)
            print(f"  {cid} {split}: n={len(y)}  AUC={roc_auc_score(y, mg):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        panel[cid] = rec
        np.savez_compressed(out_dir / f"shapley_{cid}.npz", names=np.array(rec["names"]),
                            **{f"{s}_{k}": v for s in srcs for k, v in rec[s].items()})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not panel:
        return
    ys = {s: next(iter(panel.values()))[s]["y"] for s in srcs}
    for cid, r in panel.items():
        for s in srcs:
            assert np.array_equal(r[s]["y"], ys[s]), f"{cid}: {s} label order differs across models"

    # ---- reliability matrix (val only)
    rows = []
    for cid, r in panel.items():
        phi = r["val"]["phi"]; absphi = np.abs(phi)
        share = absphi / np.maximum(absphi.sum(1, keepdims=True), 1e-12)
        dom = np.argmax(absphi, 1)
        for k, ch in enumerate(r["names"]):
            auc = roc_auc_score(ys["val"], phi[:, k]) if np.std(phi[:, k]) > 0 else np.nan
            rng = np.random.default_rng(k); boots = []
            for _ in range(500):
                i = rng.integers(0, len(phi), len(phi))
                if len(set(ys["val"][i])) == 2 and np.std(phi[i, k]) > 0:
                    boots.append(roc_auc_score(ys["val"][i], phi[i, k]))
            stab = (X.spearman(phi[:len(r["val"]["phi_noisy"]), k], r["val"]["phi_noisy"][:, k])
                    if "phi_noisy" in r["val"] else np.nan)
            others = [(c2, r2["names"].index(ch)) for c2, r2 in panel.items()
                      if c2 != cid and ch in r2["names"]]
            agree = (np.mean([np.mean(np.sign(phi[:, k]) == np.sign(panel[c2]["val"]["phi"][:, k2]))
                              for c2, k2 in others]) if others else np.nan)
            rows.append({"model": cid, "channel": ch, "mean_share": float(share[:, k].mean()),
                         "dominant_coverage": float((dom == k).mean()),
                         "phi_auc": float(auc),
                         "phi_auc_CI": (f"[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
                                        if boots else "n/a"),
                         "stability_rho": float(stab), "sign_agreement_with_panel": float(agree)})
    rel = pd.DataFrame(rows)
    rel.to_csv(out_dir / "reliability_matrix.csv", index=False)

    # ---- calibrated single models, then two stacks — everything fitted on val
    def platt(mv, yv, mt):
        return LogisticRegression(C=1e6).fit(mv[:, None], yv).predict_proba(mt[:, None])[:, 1]

    preds = {f"{cid} (Platt)": platt(r["val"]["margin"], ys["val"], r["test"]["margin"])
             for cid, r in panel.items()}
    val_nll = {cid: cv_platt_nll(r["val"]["margin"], ys["val"]) for cid, r in panel.items()}
    val_auc = {cid: float(roc_auc_score(ys["val"], r["val"]["margin"])) for cid, r in panel.items()}
    best_id = min(val_nll, key=val_nll.get)
    baselines = {"best single (val CV-NLL)": f"{best_id} (Platt)"}
    if PREREGISTERED_BASELINE in panel:
        baselines["pre-registered B0"] = f"{PREREGISTERED_BASELINE} (Platt)"

    def stack(feat_fn, name):
        Fv = np.hstack([feat_fn(r["val"]) for r in panel.values()])
        Ft = np.hstack([feat_fn(r["test"]) for r in panel.values()])
        mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
        lr = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000)
        lr.fit((Fv - mu) / sd, ys["val"])
        preds[name] = lr.predict_proba((Ft - mu) / sd)[:, 1]
        return lr.coef_[0]

    coef_m = stack(lambda d: d["margin"][:, None], "stack: margins")
    coef_p = stack(lambda d: d["phi"], "stack: branch Shapley")
    wbm, off = {}, 0
    for cid, r in panel.items():
        k = len(r["names"]); wbm[cid] = float(np.abs(coef_p[off:off + k]).sum()); off += k
    tot = sum(wbm.values()) or 1.0
    wbm = {k: v / tot for k, v in wbm.items()}
    routed_to = max(wbm, key=wbm.get)

    yt = ys["test"]
    rng = np.random.default_rng(1)
    boot_i = [i for i in (rng.integers(0, len(yt), len(yt)) for _ in range(2000))
              if len(set(yt[i])) == 2]
    comp = []
    for name, p in preds.items():
        row = {"model": name, "test_AUC": roc_auc_score(yt, p),
               "AUC_CI": "[{:.4f}, {:.4f}]".format(*np.percentile(
                   [roc_auc_score(yt[i], p[i]) for i in boot_i], [2.5, 97.5])),
               "Brier": brier_score_loss(yt, p), "NLL": log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6)),
               "ECE_equal_mass": X.ece_equal_mass(yt, p)}
        for tag, base in baselines.items():
            if name == base:
                continue
            short = "best" if tag.startswith("best") else "B0"
            da = np.array([roc_auc_score(yt[i], p[i]) - roc_auc_score(yt[i], preds[base][i]) for i in boot_i])
            db = np.array([np.mean((p[i] - yt[i]) ** 2) - np.mean((preds[base][i] - yt[i]) ** 2)
                           for i in boot_i])
            lo, hi = np.percentile(da, [2.5, 97.5]); blo, bhi = np.percentile(db, [2.5, 97.5])
            mda, mdb = float(np.mean(da)), float(np.mean(db))
            better = (((lo > 0 and mda >= MIN_DAUC) or (bhi < 0 and -mdb >= MIN_DBRIER))
                      and not (hi < 0 and -mda >= MIN_DAUC) and not (blo > 0 and mdb >= MIN_DBRIER))
            row.update({f"dAUC_vs_{short}": mda, f"dAUC_CI_vs_{short}": f"[{lo:+.4f}, {hi:+.4f}]",
                        f"dBrier_vs_{short}": mdb, f"dBrier_CI_vs_{short}": f"[{blo:+.4f}, {bhi:+.4f}]",
                        f"resolved_better_than_{short}": bool(better)})
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(out_dir / "arbitration_comparison.csv", index=False)
    stacks = comp[comp["model"].str.startswith("stack")]
    col = "resolved_better_than_best"
    wins = (stacks[stacks[col].fillna(False).astype(bool)] if col in stacks else stacks.iloc[0:0])
    decision = (f"ship '{wins.iloc[0]['model']}' — it beats the best single model by the pre-set margin, CI excluding 0"
                if len(wins) else
                f"ship the best single model ({best_id}, Platt-calibrated) — no stack beats it by the pre-set "
                f"margin (dAUC >= {MIN_DAUC} or dBrier <= -{MIN_DBRIER}, 95% CI excluding 0)")
    summ = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args),
            "n_val": int(len(ys["val"])), "n_test": int(len(yt)),
            "val_auc_raw": val_auc, "val_cv_nll_platt": val_nll, "best_single_by_val": best_id,
            "baselines": baselines, "stack_weight_share_by_model": wbm,
            "stack_routes_most_weight_to": routed_to,
            "preregistered_routing_prediction": PREREGISTERED_ROUTING,
            "routing_prediction_held": routed_to == PREREGISTERED_ROUTING,
            "max_shapley_efficiency_residual": float(max(r[s]["resid"].max() for r in panel.values() for s in srcs)),
            "coef_margin_stack": dict(zip(panel, coef_m.tolist())), "decision": decision,
            "min_effect": {"dAUC": MIN_DAUC, "dBrier": MIN_DBRIER},
            "checkpoints": {cid: r["meta"] for cid, r in panel.items()}}
    with open(out_dir / "arbitration_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2, default=_js)
    figure_reliability(rel, out_dir)
    print(comp[["model", "test_AUC", "AUC_CI", "Brier", "NLL"]].round(4).to_string(index=False))
    print(decision)


# ==========================================================================
# figures
# ==========================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    return plt


COLORS = {"gradcam": "#1f4e6b", "ig_blur": "#2b8a8a", "ig_mean": "#7fb3b3",
          "occlusion": "#b5651d", "kshap": "#6a3d9a", "centre": "#999999", "random": "#000000"}


def figures_spatial(tab, out_dir, examples, pc, tp, fills, grid):
    plt = _plt(); fd = out_dir / "figures"; fd.mkdir(exist_ok=True)
    z = np.load(out_dir / "curves.npz"); fr = z["fractions"]
    fig, ax = plt.subplots(2, len(fills), figsize=(2.6 * len(fills), 5.0),
                           sharex=True, sharey=True, squeeze=False)
    for c, f in enumerate(fills):
        for r, (name, k) in enumerate((("deletion", 0), ("insertion", 1))):
            a = ax[r, c]
            for m, col in COLORS.items():
                key = f"{m}__{f}"
                if key not in z:
                    continue
                mu = z[key][:, k].mean(0)
                ctrl = m in ("random", "centre")
                a.plot(fr, mu, "--" if ctrl else "-", color=col, lw=1.0 if ctrl else 1.3, label=m,
                       zorder=3 if ctrl else 2)
                # No CI bands: they span clip-to-clip spread in p0 and bury the curves.
                # The inferential comparison is fig_gain_vs_random.png (paired, per clip).
            a.set_title(f"{name} — {f} fill", fontsize=8); a.set_ylim(0, 1.02)
            if r == 1: a.set_xlabel("fraction of units")
            if c == 0: a.set_ylabel("p(predicted class)")
    ax[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(fd / "fig_deletion_insertion.png"); plt.close(fig)

    real = [m for m in METHODS + ["centre"] if m in set(tab["method"])]
    marks = {"blur": "o", "mean": "s", "roll": "^"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for off, f in zip(np.linspace(-0.2, 0.2, len(fills)), fills):
        s = tab[tab["fill"] == f].set_index("method")
        for i, m in enumerate(real):
            mu = s.loc[m, "d_del_vs_random"]
            lo, hi = [float(x) for x in s.loc[m, "d_del_CI"].strip("[]").split(",")]
            ax.errorbar(i + off, mu, yerr=[[mu - lo], [hi - mu]], fmt=marks[f], color=COLORS[m],
                        capsize=2, ms=4, mfc=COLORS[m] if f == "blur" else "white")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(real))); ax.set_xticklabels(real)
    ax.set_ylabel("deletion-AUC gain over random (95% CI)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="0.3", marker=marks[f], ls="none", ms=4,
                              mfc="0.3" if f == "blur" else "white", label=f"{f} fill") for f in fills],
              frameon=False, fontsize=7, ncol=len(fills), loc="upper left")
    fig.tight_layout(); fig.savefig(fd / "fig_gain_vs_random.png"); plt.close(fig)

    picks = []
    for lab, ok, pick_max, desc in ((1, 1, True, "confident correct fake"),
                                    (0, 1, False, "confident correct real"),
                                    (0, 0, True, "worst false positive"),
                                    (1, 0, False, "worst false negative")):
        s = pc[(pc["label"] == lab) & (pc["correct"] == ok)]
        if len(s):
            row = s.loc[s["p_fake"].idxmax() if pick_max else s["p_fake"].idxmin()]
            picks.append((row["clip_id"], desc, row["p_fake"]))
    ms = [m for m in ("gradcam", "ig_blur", "occlusion", "kshap") if m in next(iter(examples.values()))["units"]]
    if picks:
        fig, ax = plt.subplots(len(picks), len(ms) + 1, figsize=(1.5 * (len(ms) + 1), 1.6 * len(picks)),
                               squeeze=False)
        for r, (cl, desc, pf) in enumerate(picks):
            img = examples[cl]["frame"]
            ax[r, 0].imshow(img); ax[r, 0].set_ylabel(desc, fontsize=6)
            ax[r, 0].set_title(f"p(fake)={pf:.2f}", fontsize=7)
            for c, m in enumerate(ms):
                ax[r, c + 1].imshow(img)
                ax[r, c + 1].imshow(examples[cl]["units"][m].reshape(grid, grid), cmap="Blues", alpha=.6,
                                    extent=(0, img.shape[1], img.shape[0], 0), interpolation="nearest")
                if r == 0: ax[r, c + 1].set_title(m, fontsize=7)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(fd / "fig_example_maps.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ks = sorted({int(c.split("@")[1]) for c in tp.columns if c.startswith("occ@")})
    for p, lab, col in (("occ", "occlusion", COLORS["occlusion"]), ("att", "attention", COLORS["gradcam"]),
                        ("rnd", "random", "k")):
        ax[0].plot(ks, [tp[f"{p}@{k}"].mean() for k in ks], "o-", color=col, label=lab)
    ax[0].set_xlabel("frames frozen"); ax[0].set_ylabel("margin removed")
    ax[0].legend(frameon=False); ax[0].set_title("temporal faithfulness")
    edges = np.r_[0, np.sort(X.SPECTRAL_GRID), 1.0]
    for lab, col in ((1, "#b5651d"), (0, "#1f4e6b")):
        v = pc.loc[(pc["p_fake"] >= .5) == bool(lab), "spectral_cf_nyq"].dropna()
        if len(v):
            ax[1].hist(v, bins=edges, alpha=.6, color=col, label=f"pred. {'fake' if lab else 'real'} (n={len(v)})")
    ax[1].set_xlabel("low-pass cutoff that flips the verdict (× Nyquist)")
    ax[1].set_title("spectral counterfactual"); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(fd / "fig_temporal_spectral.png"); plt.close(fig)


def figure_reliability(rel, out_dir):
    plt = _plt(); fd = out_dir / "figures"; fd.mkdir(exist_ok=True)
    chans = ["spatial", "frequency", "srm", "tsc", "lmf"]
    models = list(dict.fromkeys(rel["model"]))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a, col, title in ((ax[0], "mean_share", "mean share of |φ|"), (ax[1], "phi_auc", "AUC of φ (val)")):
        M = np.full((len(models), len(chans)), np.nan)
        for r in rel.itertuples():
            M[models.index(r.model), chans.index(r.channel)] = getattr(r, col)
        a.imshow(M, cmap="Blues", vmin=0 if col == "mean_share" else 0.5, vmax=1)
        for i in range(len(models)):
            for j in range(len(chans)):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if M[i, j] > .75 else "black")
        a.set_xticks(range(len(chans))); a.set_xticklabels(chans, fontsize=7)
        a.set_yticks(range(len(models))); a.set_yticklabels(models); a.set_title(title)
    fig.tight_layout(); fig.savefig(fd / "fig_reliability_matrix.png"); plt.close(fig)


# ==========================================================================
# report — assembled from whatever is on disk, so stages can run separately
# ==========================================================================

def write_report(out_dir, demo):
    L = [f"# Explanation benchmark — report built {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if demo:
        L += ["> **DEMO RUN — random weights, synthetic clips. Plumbing check only. "
              "No number in this file describes the detector.**", ""]
    sp = out_dir / "spatial_summary.json"
    if sp.exists():
        s = json.load(open(sp))
        a = s["args"]
        tab = pd.read_csv(out_dir / "spatial_faithfulness.csv")
        L += [f"## 1. Spatial faithfulness — {a['model']}, n={s['n']}"
              + (" (PARTIAL — run was interrupted)" if s["partial"] else ""), "",
              f"Run {s['run_utc']}; T={a['frames']}, {a['size']}px, grid {a['grid']}×{a['grid']}, "
              f"IG {a['ig_steps']} steps, KernelSHAP {a['kshap'] or 'off'}, "
              f"{s['forwards_per_clip']:.0f} forwards/clip.", "",
              "Sample (errors over-sampled; population means are inverse-probability weighted):", "",
              "| stratum | population | sampled |", "|---|---|---|",
              *[f"| {k} | {v['population']} | {v['sampled']} |" for k, v in s["sample"].items()], "",
              "`d_del_vs_random` > 0 means the method's deletion curve falls faster than random's. "
              "`beats_random` requires the 95% CI to exclude zero.", "",
              _md(tab[["method", "fill", "del_auc", "ins_auc", "comp_mean", "suff_abs", "flip_frac",
                       "d_del_vs_random", "d_del_CI", "p_del", "d_del_vs_centre", "p_centre",
                       "beats_random", "borda"]]), "",
              f"**Winner** (beats random under both blur and mean-face fills, highest Borda): "
              f"**{s['winner'] or 'none — no method beats random under both primary fills'}**.", "",
              f"Fill agreement (Kendall τ of method order): {json.dumps(s['fill_rank_agreement_kendall_tau'])}.", "",
              f"Controls: {_kv(s['controls'])}", "", f"Modality gate: {_kv(s['modality_gate'])}", "",
              f"Degenerate (constant) map rate — these clips get a random ranking: "
              f"{_kv(s['degenerate_map_rate'])}", "",
              f"IG completeness residual (median; above ~0.1 the path is under-resolved — raise "
              f"--ig-steps): {_kv(s['ig_completeness_residual_median'])}", ""]
        for title, fn in (("2. Stability (Spearman ρ / top-10% Jaccard; 1 = identical)", "stability.csv"),
                          ("3. Model-randomisation sanity check (ρ near 0 is a PASS)", "sanity.csv")):
            p = out_dir / fn
            txt = p.read_text().strip() if p.exists() else ""
            L += [f"## {title}", "", ("```\n" + txt + "\n```") if txt and len(txt.splitlines()) > 1
                  else "_not run_", ""]
        L += ["## 4. Temporal faithfulness and counterfactuals", "", "```",
              json.dumps(s["temporal"], indent=1, default=_js),
              json.dumps(s["counterfactuals"], indent=1, default=_js), "```", "",
              "## 5. Does faithfulness predict correctness?", "", "```",
              json.dumps(s["faithfulness_vs_correctness"], indent=1, default=_js), "```", ""]
    else:
        L += ["## 1–5. Spatial study", "", "_not run yet_", ""]
    ap = out_dir / "arbitration_summary.json"
    if ap.exists():
        a = json.load(open(ap))
        comp = pd.read_csv(out_dir / "arbitration_comparison.csv")
        L += [f"## 6. Reliability matrix (val, n={a['n_val']})", "",
              _md(pd.read_csv(out_dir / "reliability_matrix.csv").round(3)), "",
              f"## 7. Arbitration vs single models (test, n={a['n_test']})", "",
              _md(comp.round(4)), "",
              f"Best single model chosen on val by 5-fold CV NLL of its Platt calibrator: "
              f"**{a['best_single_by_val']}** (CV NLL {_kv(a['val_cv_nll_platt'])}).", "",
              f"Stack weight share by model: {_kv(a['stack_weight_share_by_model'])}. "
              f"Pre-registered prediction (cfx_implementation.md §4): most weight to "
              f"{a['preregistered_routing_prediction']} — **{'held' if a['routing_prediction_held'] else 'did not hold'}** "
              f"(observed: {a['stack_routes_most_weight_to']}).", "", f"**Decision: {a['decision']}**", ""]
    else:
        L += ["## 6–7. Reliability matrix and arbitration", "", "_not run yet_", ""]
    L += ["## Not covered here", "",
          "* Narration quality (groundedness, reading grade): `python scripts/cfx_eval.py --n 100 --narrate`.",
          "* Over-trust of the plain-language register: needs people (cfx_implementation.md §11).", ""]
    (out_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def _kv(d):
    return ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def _nanmean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _med(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="C3")
    ap.add_argument("--stages", default="spatial,arbitration")
    ap.add_argument("--n", type=int, default=200, help="test clips for the spatial study")
    ap.add_argument("--n-stability", type=int, default=50)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--kshap", type=int, default=512,
                    help="coalitions over the 64 units; 0 disables KernelSHAP")
    ap.add_argument("--batch", type=int, default=4, help="clips per forward (x16 frames)")
    ap.add_argument("--ig-batch", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="fp16 forwards (~2x faster; scores drift ~1e-3)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lmf-cache", default=str(ROOT / "data" / "processed" / "lmf_cache_16f.npz"))
    ap.add_argument("--arb-limit", type=int, default=0, help="cap clips per split (0 = all)")
    ap.add_argument("--arb-noise-n", type=int, default=200)
    ap.add_argument("--cpu-fallback", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--resummarise", action="store_true",
                    help="rebuild stage-1 tables/figures and the report from disk, no model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.cpu_fallback or device.type == "cpu") and not args.demo:
        args.n, args.n_stability = min(args.n, 32), min(args.n_stability, 8)
        args.kshap, args.ig_steps, args.batch, args.ig_batch = 0, 8, 2, 1
        print("CPU fallback: n=32, stability n=8, KernelSHAP off, IG 8 steps")
    if args.demo:
        args.frames, args.size, args.ig_steps, args.ig_batch = 4, 64, 8, 2
        args.n, args.n_stability, args.kshap = min(args.n, 12), min(args.n_stability, 3), 64
        args.arb_limit, args.arb_noise_n = args.arb_limit or 80, 40
    out_dir = Path(args.out or ROOT / "results" / ("xai_demo" if args.demo else "xai"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   out: {out_dir}")
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()
    stages = set(args.stages.split(","))
    if args.resummarise:
        if (out_dir / "_stage1_raw.pkl").exists():
            finish_spatial(out_dir)
    else:
        if "spatial" in stages:
            stage_spatial(args, device, out_dir)
        if "arbitration" in stages:
            stage_arbitration(args, device, out_dir)
    write_report(out_dir, args.demo)
    print(f"\nwrote {out_dir / 'REPORT.md'}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
