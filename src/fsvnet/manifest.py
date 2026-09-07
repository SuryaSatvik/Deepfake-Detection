"""
manifest.py
===========

Turns a downloaded dataset into one CSV of (video_path, label, identity_group,
source_dataset, method) rows, then splits it into train/val/test.

THE SPLIT IS THE MOST IMPORTANT CODE IN THIS PROJECT
----------------------------------------------------
The previous image-level project in this capstone reported 0.7998 AUC on a
random 80/20 split and 0.5702 on a genuinely held-out partition, because 17.7%
of its "test" rows had a near-duplicate in train. Video data makes that failure
mode far worse:

  * one video yields many near-identical frames;
  * one person appears in dozens of videos;
  * a face-swap clip shares its *background and target identity* with the real
    clip it was derived from.

So a random split over clips, or even over videos, leaks. The only honest unit
is the **identity group**, and a swap `id0 -> id16` binds those two identities
together — put `id0_id16_0000.mp4` in train and any `id16_*.mp4` in test and
the model has already seen that face.

This module therefore builds identity groups with a union-find over every
identity pair a manipulation touches, and splits over the resulting connected
components. `split_by="random"` exists only so you can *measure* the leak and
report the gap; never use it for a headline number.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .datasets_fetch import VIDEO_EXTS
from .utils import log, save_json

COLUMNS = ["video_path", "label", "label_name", "identity", "group",
           "dataset", "method", "split"]


# --------------------------------------------------------------------------
# Union-find over identities
# --------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# --------------------------------------------------------------------------
# Per-dataset adapters -> list of dicts with identities as a list
# --------------------------------------------------------------------------

_CELEB_REAL = re.compile(r"^id(\d+)_(\d+)$")
_CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d+)$")
_FFPP_FAKE = re.compile(r"^(\d+)_(\d+)$")
# A "subject" token in a flat folder: real_s0007_c2.mp4 / fake_smooth_s0007_c2.mp4
# both belong to identity s0007. Name your own files this way and you get
# identity-disjoint splits for free.
_SUBJECT = re.compile(r"(?:^|_)(s\d+)(?:_|$)")


def _rows_celebdf(root: Path) -> List[dict]:
    rows = []
    specs = [("Celeb-real", 0, "real"),
             ("YouTube-real", 0, "real"),
             ("Celeb-synthesis", 1, "celebdf_swap")]
    for sub, label, method in specs:
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            stem = p.stem
            ids: List[str] = []
            m = _CELEB_FAKE.match(stem)
            if m:
                ids = [f"celeb_id{m.group(1)}", f"celeb_id{m.group(2)}"]
            else:
                m = _CELEB_REAL.match(stem)
                if m:
                    ids = [f"celeb_id{m.group(1)}"]
                else:
                    # YouTube-real/00000.mp4 — each clip is its own identity
                    ids = [f"celeb_yt_{stem}"]
            rows.append(dict(video_path=str(p), label=label,
                             label_name="fake" if label else "real",
                             identities=ids, dataset="celebdf", method=method))
    return rows


def _rows_ffpp(root: Path) -> List[dict]:
    rows = []
    orig = root / "original_sequences"
    for p in sorted(orig.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        comp = p.parent.parent.name          # c23 / c40 / raw
        rows.append(dict(video_path=str(p), label=0, label_name="real",
                         identities=[f"ffpp_{p.stem}"], dataset="ffpp",
                         method=f"original_{comp}"))

    man = root / "manipulated_sequences"
    if man.exists():
        for method_dir in sorted(d for d in man.iterdir() if d.is_dir()):
            for p in sorted(method_dir.rglob("*")):
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue
                m = _FFPP_FAKE.match(p.stem)
                ids = ([f"ffpp_{m.group(1)}", f"ffpp_{m.group(2)}"] if m
                       else [f"ffpp_{p.stem}"])
                comp = p.parent.parent.name
                rows.append(dict(video_path=str(p), label=1, label_name="fake",
                                 identities=ids, dataset="ffpp",
                                 method=f"{method_dir.name}_{comp}"))
    return rows


def _rows_flat(root: Path, dataset: str) -> List[dict]:
    """real/ + fake/ layout (DFDC after sorting, or your own folder)."""
    rows = []
    meta_path = root / "metadata.json"
    original_of: Dict[str, str] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for k, v in meta.items():
                if isinstance(v, dict) and v.get("original"):
                    original_of[Path(k).stem] = Path(str(v["original"])).stem
        except Exception as e:
            log.warning(f"could not parse {meta_path}: {e}")

    for sub, label in [("real", 0), ("fake", 1)]:
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            m = _SUBJECT.search(p.stem)
            ids = [f"{dataset}_{m.group(1)}" if m else f"{dataset}_{p.stem}"]
            src = original_of.get(p.stem)
            if src:
                ids.append(f"{dataset}_{src}")     # bind fake to its source
            rows.append(dict(video_path=str(p), label=label,
                             label_name="fake" if label else "real",
                             identities=ids, dataset=dataset, method=sub))
    return rows


ADAPTERS = {
    "celebdf": _rows_celebdf,
    "ffpp": _rows_ffpp,
    "dfdcp": lambda r: _rows_flat(r, "dfdcp"),
    "custom": lambda r: _rows_flat(r, "custom"),
    "synthetic": lambda r: _rows_flat(r, "synthetic"),
}


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_manifest(sources: Dict[str, Path]) -> pd.DataFrame:
    """
    sources: {"celebdf": Path(...), "ffpp": Path(...), ...}
    Returns a DataFrame with one row per video and a `group` column holding the
    connected-component id of its identity group.
    """
    rows: List[dict] = []
    for kind, root in sources.items():
        if root is None:
            continue
        fn = ADAPTERS.get(kind)
        if fn is None:
            log.warning(f"no adapter for dataset '{kind}' — skipping")
            continue
        got = fn(Path(root))
        log.info(f"{kind}: {len(got)} videos")
        rows.extend(got)

    if not rows:
        raise RuntimeError(
            "No videos found. Check the roots you set on cfg.paths, or run "
            "fsvnet.synthetic.build_synthetic_dataset(cfg) to generate a "
            "pipeline smoke set."
        )

    # union-find: every manipulation binds the identities it touches
    uf = _UF()
    for r in rows:
        ids = r["identities"]
        for i in ids:
            uf.find(i)
        for i in ids[1:]:
            uf.union(ids[0], i)

    for r in rows:
        r["identity"] = r["identities"][0]
        r["group"] = uf.find(r["identities"][0])

    df = pd.DataFrame(rows).drop(columns=["identities"])
    df["split"] = ""
    n_groups = df["group"].nunique()
    log.info(f"manifest: {len(df)} videos, {n_groups} identity groups, "
             f"{int((df.label == 0).sum())} real / {int((df.label == 1).sum())} fake")
    if n_groups < 10:
        log.warning(f"only {n_groups} identity groups — an identity-disjoint "
                    "split will be very coarse. Expect noisy val/test numbers.")
    return df


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _assign_groups(df: pd.DataFrame, fracs: Tuple[float, float, float],
                   seed: int, key: str) -> pd.DataFrame:
    """
    Greedy balanced assignment over groups: shuffle groups, then repeatedly
    drop the next group into whichever split is furthest below its quota.
    Keeps the real/fake ratio from drifting when group sizes are uneven.
    """
    rng = np.random.default_rng(seed)
    sizes = df.groupby(key).size().to_dict()
    groups = list(sizes.keys())
    rng.shuffle(groups)

    names = ["train", "val", "test"]
    quota = {n: f * len(df) for n, f in zip(names, fracs)}
    filled = {n: 0.0 for n in names}
    assign: Dict[str, str] = {}

    # largest groups first so they land where they still fit
    for g in sorted(groups, key=lambda g: -sizes[g]):
        deficit = {n: (quota[n] - filled[n]) / max(quota[n], 1e-9) for n in names}
        target = max(names, key=lambda n: deficit[n])
        assign[g] = target
        filled[target] += sizes[g]

    df = df.copy()
    df["split"] = df[key].map(assign)
    return df


def make_splits(df: pd.DataFrame, split_by: str = "identity",
                fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                seed: int = 42,
                celebdf_official_test: Optional[Path] = None) -> pd.DataFrame:
    """
    split_by:
      "identity" — connected identity components (DEFAULT, the honest one)
      "video"    — disjoint videos, identities may recur (leaks, but less)
      "random"   — disjoint rows (LEAKS BADLY; for measuring the gap only)

    If `celebdf_official_test` points at Celeb-DF's List_of_testing_videos.txt,
    those videos are forced into test and everything else is split over the
    remaining groups — so your number is comparable to published ones.
    """
    if split_by == "identity":
        out = _assign_groups(df, fracs, seed, "group")
    elif split_by == "video":
        out = _assign_groups(df, fracs, seed, "video_path")
    elif split_by == "random":
        log.warning("split_by='random' LEAKS. Use it only to quantify the leak.")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_tr = int(fracs[0] * len(df))
        n_va = int(fracs[1] * len(df))
        out = df.copy()
        out.loc[df.index[idx[:n_tr]], "split"] = "train"
        out.loc[df.index[idx[n_tr:n_tr + n_va]], "split"] = "val"
        out.loc[df.index[idx[n_tr + n_va:]], "split"] = "test"
    else:
        raise ValueError(f"unknown split_by: {split_by}")

    if celebdf_official_test and Path(celebdf_official_test).exists():
        names = set()
        for line in Path(celebdf_official_test).read_text().splitlines():
            parts = line.strip().split()
            if parts:
                names.add(Path(parts[-1]).name)
        mask = out["video_path"].map(lambda p: Path(p).name in names)
        n = int(mask.sum())
        if n:
            # move the whole identity group, not just the listed videos
            groups = set(out.loc[mask, "group"])
            out.loc[out["group"].isin(groups), "split"] = "test"
            log.info(f"applied Celeb-DF official test list: {n} listed videos, "
                     f"{int(out['group'].isin(groups).sum())} videos moved to test "
                     "(whole identity groups)")

    _report_split(out)
    return out


def _report_split(df: pd.DataFrame) -> None:
    print("\nSplit summary")
    print("-" * 62)
    for s in ["train", "val", "test"]:
        sub = df[df.split == s]
        if not len(sub):
            continue
        r, f = int((sub.label == 0).sum()), int((sub.label == 1).sum())
        print(f"  {s:<6} {len(sub):>6} videos   real {r:>5}  fake {f:>5}  "
              f"({f / max(1, len(sub)) * 100:4.1f}% fake)  "
              f"groups {sub['group'].nunique():>4}")
    # the check that actually matters
    overlap = _group_overlap(df)
    print("-" * 62)
    if overlap:
        print(f"  !! IDENTITY GROUP OVERLAP: {overlap}  <- your split leaks")
    else:
        print("  OK  no identity group appears in more than one split")
    print()


def _group_overlap(df: pd.DataFrame) -> Dict[str, int]:
    per = defaultdict(set)
    for g, s in zip(df["group"], df["split"]):
        per[g].add(s)
    bad = {g: len(ss) for g, ss in per.items() if len(ss) > 1}
    return {"n_groups_in_multiple_splits": len(bad)} if bad else {}


def leakage_audit(df: pd.DataFrame) -> dict:
    """
    Quantify what a random split would have bought you, so the report can state
    it as a measured number rather than an assertion.
    """
    from copy import deepcopy
    rnd = make_splits(deepcopy(df), split_by="random")
    tr = set(rnd.loc[rnd.split == "train", "group"])
    te_rows = rnd[rnd.split == "test"]
    shared = int(te_rows["group"].isin(tr).sum())
    audit = {
        "random_split_test_videos": int(len(te_rows)),
        "test_videos_whose_identity_is_in_train": shared,
        "contamination_rate": round(shared / max(1, len(te_rows)), 4),
    }
    print(f"\nLeakage audit (random split): {audit['contamination_rate'] * 100:.1f}% "
          f"of test videos share an identity group with train.")
    print("That is the number a random-split AUC would be measuring.\n")
    return audit


# --------------------------------------------------------------------------

def save_manifest(df: pd.DataFrame, out_dir: Path, name: str = "manifest") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.csv"
    df.to_csv(p, index=False)
    stats = {
        "n_videos": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "by_split": df.groupby("split").size().to_dict(),
        "by_dataset": df.groupby("dataset").size().to_dict(),
        "by_method": df.groupby("method").size().to_dict(),
        "label_balance": df.groupby("label").size().to_dict(),
    }
    save_json(stats, out_dir / f"{name}_stats.json")
    log.info(f"manifest saved: {p}")
    return p


def load_manifest(path) -> pd.DataFrame:
    return pd.read_csv(path)


__all__ = ["build_manifest", "make_splits", "save_manifest", "load_manifest",
           "leakage_audit", "ADAPTERS", "COLUMNS"]
