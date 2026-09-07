"""
prepare_data.py
===============

Command-line entry point for everything that happens before training:
acquisition guidance, layout verification, manifest building, identity-disjoint
splitting, and the one-off preprocessing cache.

Usage
-----
    # 0. what do I need to download, and from where?
    python scripts/prepare_data.py guide

    # 1. did my download land in the right shape?
    python scripts/prepare_data.py verify --celebdf "D:/data/Celeb-DF-v2"

    # 2. no dataset yet — make the synthetic smoke set
    python scripts/prepare_data.py synthetic --n 100

    # 3. build the manifest + splits (identity-disjoint by default)
    python scripts/prepare_data.py manifest --celebdf "D:/data/Celeb-DF-v2"

    # 4. decode, crop and cache every clip (the slow one — run it once)
    python scripts/prepare_data.py preprocess

    # 5. sort a DFDC part into real/ and fake/ first
    python scripts/prepare_data.py dfdc-sort --part "D:/dfdc/part_0" --dest "D:/dfdc/sorted"

Everything writes into data/manifests/ and data/clips/ and is idempotent — a
re-run skips work that is already done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet.config import Config  # noqa: E402
from fsvnet.utils import log  # noqa: E402


def _cfg_from_args(a) -> Config:
    cfg = Config()
    if getattr(a, "celebdf", None):
        cfg.paths.celebdf_root = Path(a.celebdf)
    if getattr(a, "ffpp", None):
        cfg.paths.ffpp_root = Path(a.ffpp)
    if getattr(a, "dfdc", None):
        cfg.paths.dfdc_root = Path(a.dfdc)
    if getattr(a, "custom", None):
        cfg.paths.custom_root = Path(a.custom)
    if getattr(a, "frames", None):
        cfg.data.num_frames = a.frames
    if getattr(a, "img_size", None):
        cfg.data.img_size = a.img_size
    if getattr(a, "face_backend", None):
        cfg.data.face_backend = a.face_backend
    cfg.paths.mkdirs()
    return cfg


def _sources(cfg: Config) -> dict:
    src = {}
    for kind, root in [("celebdf", cfg.paths.celebdf_root),
                       ("ffpp", cfg.paths.ffpp_root),
                       ("dfdcp", cfg.paths.dfdc_root),
                       ("custom", cfg.paths.custom_root)]:
        if root:
            src[kind] = Path(root)
    syn = Path(cfg.paths.data_dir) / "synthetic"
    if not src and syn.exists():
        log.info("no dataset roots given — using the synthetic set")
        src["synthetic"] = syn
    return src


# --------------------------------------------------------------------------

def cmd_guide(a) -> int:
    from fsvnet.datasets_fetch import print_acquisition_guide
    print_acquisition_guide()
    return 0


def cmd_verify(a) -> int:
    from fsvnet.datasets_fetch import verify
    cfg = _cfg_from_args(a)
    any_ok = False
    for kind, root in _sources(cfg).items():
        rep = verify(root, kind)
        any_ok |= bool(rep["ok"])
    if not any_ok:
        print("\nNothing verified OK. Pass --celebdf / --ffpp / --dfdc / --custom "
              "with the folder you downloaded into.")
    return 0 if any_ok else 1


def cmd_synthetic(a) -> int:
    from fsvnet.synthetic import build_synthetic_dataset, controlled_tsc_test
    cfg = _cfg_from_args(a)
    build_synthetic_dataset(cfg, n_per_class=a.n, T=a.clip_len, size=a.size)
    if not a.no_test:
        controlled_tsc_test(n=20, T=32, size=224)
    return 0


def cmd_manifest(a) -> int:
    from fsvnet.manifest import (build_manifest, leakage_audit, make_splits,
                                 save_manifest)
    cfg = _cfg_from_args(a)
    src = _sources(cfg)
    if not src:
        print("No sources. Run `guide`, then pass a dataset root.")
        return 1

    df = build_manifest(src)
    official = None
    if cfg.paths.celebdf_root:
        p = Path(cfg.paths.celebdf_root) / "List_of_testing_videos.txt"
        official = p if p.exists() else None

    df = make_splits(df, a.split_by,
                     (cfg.data.train_frac, cfg.data.val_frac, cfg.data.test_frac),
                     cfg.data.seed, official)
    if a.audit:
        leakage_audit(df)
    save_manifest(df, cfg.paths.manifest_dir, a.name)
    return 0


def cmd_preprocess(a) -> int:
    import pandas as pd

    from fsvnet.dataset import preprocess_dataset
    cfg = _cfg_from_args(a)
    path = Path(cfg.paths.manifest_dir) / f"{a.name}.csv"
    if not path.exists():
        print(f"{path} not found — run `manifest` first.")
        return 1

    df = pd.read_csv(path)
    df = preprocess_dataset(df, cfg, overwrite=a.overwrite, limit=a.limit)
    out = Path(cfg.paths.manifest_dir) / f"{a.name}_cached.csv"
    df.to_csv(out, index=False)
    log.info(f"cached manifest saved: {out}")
    if "ok" in df and not df["ok"].all():
        bad = df[~df.ok.astype(bool)]
        print(f"\n{len(bad)} videos failed. First few reasons:")
        for _, r in bad.head(5).iterrows():
            print(f"  {Path(str(r['video_path'])).name}: {r.get('error')}")
    return 0


def cmd_dfdc_sort(a) -> int:
    from fsvnet.datasets_fetch import sort_dfdc_by_metadata
    sort_dfdc_by_metadata(a.part, a.dest, move=a.move)
    return 0


def cmd_kaggle(a) -> int:
    from fsvnet.datasets_fetch import fetch_kaggle
    fetch_kaggle(a.slug, a.dest, competition=a.competition)
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="FSV-Net data preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_roots(p):
        p.add_argument("--celebdf"); p.add_argument("--ffpp")
        p.add_argument("--dfdc"); p.add_argument("--custom")

    p = sub.add_parser("guide", help="print the dataset acquisition checklist")
    p.set_defaults(fn=cmd_guide)

    p = sub.add_parser("verify", help="check a downloaded dataset's layout")
    add_roots(p); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("synthetic", help="generate the synthetic smoke dataset")
    p.add_argument("--n", type=int, default=100, help="clips per class")
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--no-test", action="store_true")
    p.set_defaults(fn=cmd_synthetic)

    p = sub.add_parser("manifest", help="build the manifest and splits")
    add_roots(p)
    p.add_argument("--name", default="manifest")
    p.add_argument("--split-by", default="identity",
                   choices=["identity", "video", "random"])
    p.add_argument("--audit", action="store_true",
                   help="also report what a random split would have leaked")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("preprocess", help="decode, crop and cache every clip")
    add_roots(p)
    p.add_argument("--name", default="manifest")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--face-backend", default=None,
                   choices=["auto", "mediapipe", "opencv", "none"])
    p.set_defaults(fn=cmd_preprocess)

    p = sub.add_parser("dfdc-sort", help="sort a DFDC part into real/ and fake/")
    p.add_argument("--part", required=True); p.add_argument("--dest", required=True)
    p.add_argument("--move", action="store_true")
    p.set_defaults(fn=cmd_dfdc_sort)

    p = sub.add_parser("kaggle", help="download a Kaggle dataset or competition")
    p.add_argument("--slug", required=True); p.add_argument("--dest", required=True)
    p.add_argument("--competition", action="store_true")
    p.set_defaults(fn=cmd_kaggle)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
