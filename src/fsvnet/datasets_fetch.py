"""
datasets_fetch.py
=================

Dataset acquisition for the video pipeline.

READ THIS FIRST — why there is no one-click download
----------------------------------------------------
Every standard video-deepfake benchmark is access-gated. None of them can be
pulled with a plain `wget`, and no code can sign a EULA for you:

  * FaceForensics++  — signed EULA; the authors then email you a personalised
                       `faceforensics_download_v4.py`.
  * Celeb-DF v2      — request form; the authors email you a download link.
  * DFDC (full)      — Kaggle competition rules must be accepted in a browser.
  * DFDC Preview     — Meta's dataset page, separate agreement.

So this module does three things instead:

  1. `print_acquisition_guide()` — tells you exactly what to click, in order,
     with realistic sizes and turnaround times. Start the FF++/Celeb-DF
     requests on day one; approval takes 1–3 days and it is the single most
     common reason a video capstone stalls.
  2. `fetch_kaggle(slug, dest)` — the one thing that *is* automatable, for the
     mirrors and the smaller sets, using your own `kaggle.json` token.
  3. `verify(root, kind)` — points at whatever you managed to download and
     tells you whether the layout is what the manifest builder expects,
     before you waste an hour preprocessing.

Nothing here needs a GPU and nothing here needs torch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .utils import log

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    key: str
    name: str
    access: str            # "eula" | "form" | "kaggle" | "open"
    approx_size: str
    role: str              # what we use it for
    landing: str           # where to start
    notes: str


REGISTRY: Dict[str, DatasetSpec] = {
    "celebdf": DatasetSpec(
        key="celebdf",
        name="Celeb-DF v2",
        access="form",
        approx_size="~9 GB (590 real + 5,639 fake videos)",
        role="PRIMARY train/val/test. Modern, high-quality face swaps; the "
             "standard cross-dataset yardstick.",
        landing="https://github.com/yuezunli/celeb-deepfakeforensics",
        notes="Fill the request form linked in that repo's README; the authors "
              "email a download link, usually within a few days. Expected "
              "layout: Celeb-real/, Celeb-synthesis/, YouTube-real/, "
              "List_of_testing_videos.txt — keep that file, the manifest "
              "builder uses it for the official test split.",
    ),
    "ffpp": DatasetSpec(
        key="ffpp",
        name="FaceForensics++ (c23)",
        access="eula",
        approx_size="~40 GB for c23 (1000 originals + 5 manipulations)",
        role="SECONDARY training source and cross-manipulation evaluation.",
        landing="https://github.com/ondyari/FaceForensics",
        notes="Sign the EULA linked in that repo; they email you "
              "faceforensics_download_v4.py. Then, for a laptop-sized subset:\n"
              "    python faceforensics_download_v4.py <out> -d all -c c23 -t videos -n 200\n"
              "The -n flag caps videos per manipulation — start at 200, not the "
              "full set. DF40 (NeurIPS'24) showed FF++ alone does NOT "
              "generalise to modern deepfakes, so train on it, never claim "
              "generalisation from it.",
    ),
    "dfdcp": DatasetSpec(
        key="dfdcp",
        name="DFDC Preview",
        access="kaggle",
        approx_size="~5 GB (~5,000 clips)",
        role="HELD-OUT cross-dataset test only. Never train on this if you "
             "want an honest transfer number.",
        landing="https://www.kaggle.com/c/deepfake-detection-challenge/data",
        notes="Needs a Kaggle account, accepting the competition rules in a "
              "browser, and ~/.kaggle/kaggle.json. The full DFDC is 470 GB — "
              "you want the preview or a single training part, not all of it.",
    ),
    "custom": DatasetSpec(
        key="custom",
        name="Your own folder",
        access="open",
        approx_size="whatever you have",
        role="Anything with real/ and fake/ subfolders of video files.",
        landing="-",
        notes="Layout: <root>/real/*.mp4 and <root>/fake/*.mp4. Filenames "
              "should embed a stable identity token if you want "
              "identity-disjoint splits (see manifest.py).",
    ),
}


def print_acquisition_guide() -> None:
    """Print the day-one checklist. Call this from the notebook, cell 2."""
    print("=" * 78)
    print("DATASET ACQUISITION — do the gated requests FIRST, they take days")
    print("=" * 78)
    order = ["celebdf", "ffpp", "dfdcp", "custom"]
    for i, key in enumerate(order, 1):
        s = REGISTRY[key]
        print(f"\n{i}. {s.name}   [{s.access.upper()}]   {s.approx_size}")
        print(f"   role  : {s.role}")
        print(f"   start : {s.landing}")
        for line in s.notes.split("\n"):
            print(f"   note  : {line}")
    print("\n" + "-" * 78)
    print("MINIMUM VIABLE PATH if nothing has arrived yet:")
    print("  from fsvnet.synthetic import build_synthetic_dataset")
    print("  build_synthetic_dataset(cfg)     # 200 clips, ~1 min, no download")
    print("  ...which validates the whole pipeline end to end, and doubles as")
    print("  the unit test for the TSC branch (see synthetic.py).")
    print("-" * 78)


# --------------------------------------------------------------------------
# Kaggle
# --------------------------------------------------------------------------

def kaggle_available() -> bool:
    if shutil.which("kaggle") is None:
        return False
    token = Path.home() / ".kaggle" / "kaggle.json"
    return token.exists() or "KAGGLE_USERNAME" in os.environ


def fetch_kaggle(slug: str, dest: os.PathLike, competition: bool = False,
                 unzip: bool = True) -> Path:
    """
    Download a Kaggle dataset or competition into `dest`.

    Parameters
    ----------
    slug : "owner/dataset-name", or the competition name if competition=True.
    competition : use `kaggle competitions download` instead of `datasets`.

    Requires the kaggle CLI and a token at ~/.kaggle/kaggle.json
    (Kaggle → your profile → Settings → API → Create New Token).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not kaggle_available():
        raise RuntimeError(
            "kaggle CLI or credentials not found.\n"
            "  pip install kaggle\n"
            "  then put kaggle.json in %USERPROFILE%\\.kaggle\\ (Windows) "
            "or ~/.kaggle/ (Linux/mac)\n"
            "Get the token from Kaggle → Settings → API → Create New Token."
        )

    cmd = ["kaggle", "competitions" if competition else "datasets", "download",
           "-c" if competition else "-d", slug, "-p", str(dest)]
    log.info("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    if unzip:
        for z in dest.glob("*.zip"):
            log.info(f"unzipping {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()
    return dest


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _count_videos(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)


def verify(root: os.PathLike, kind: str) -> Dict[str, object]:
    """
    Check that a downloaded dataset has the layout the manifest builder wants.
    Returns a report dict and prints a readable summary. Never raises — the
    point is to tell you what is wrong, not to stop the notebook.
    """
    root = Path(root)
    report: Dict[str, object] = {"kind": kind, "root": str(root),
                                 "exists": root.exists(), "ok": False,
                                 "problems": [], "counts": {}}
    if not root.exists():
        report["problems"].append(f"path does not exist: {root}")
        _print_report(report)
        return report

    if kind == "celebdf":
        expected = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
        for d in expected:
            n = _count_videos(root / d)
            report["counts"][d] = n
            if n == 0:
                report["problems"].append(f"missing or empty: {root / d}")
        tl = root / "List_of_testing_videos.txt"
        report["counts"]["List_of_testing_videos.txt"] = tl.exists()
        if not tl.exists():
            report["problems"].append(
                "List_of_testing_videos.txt not found — without it you lose the "
                "official test split and must fall back to identity-disjoint "
                "splitting (still fine, just not comparable to published numbers)."
            )

    elif kind == "ffpp":
        orig = root / "original_sequences" / "youtube" / "c23" / "videos"
        report["counts"]["original(c23)"] = _count_videos(orig)
        if report["counts"]["original(c23)"] == 0:
            # tolerate the flatter layouts people end up with
            alt = _count_videos(root / "original_sequences")
            report["counts"]["original(any)"] = alt
            if alt == 0:
                report["problems"].append("no original_sequences videos found")
        man = root / "manipulated_sequences"
        methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
        for m in methods:
            n = _count_videos(man / m)
            report["counts"][m] = n
        if sum(report["counts"].get(m, 0) for m in methods) == 0:
            report["problems"].append("no manipulated_sequences videos found")

    elif kind in ("dfdcp", "custom"):
        real = _count_videos(root / "real")
        fake = _count_videos(root / "fake")
        report["counts"] = {"real": real, "fake": fake}
        if real == 0 or fake == 0:
            loose = _count_videos(root)
            report["counts"]["loose_videos"] = loose
            report["problems"].append(
                f"expected {root}/real and {root}/fake; found {loose} videos "
                "elsewhere. For DFDC, use the metadata.json to sort clips into "
                "real/ and fake/ first (see sort_dfdc_by_metadata)."
            )
    else:
        report["problems"].append(f"unknown dataset kind: {kind}")

    report["ok"] = len(report["problems"]) == 0
    _print_report(report)
    return report


def _print_report(r: Dict[str, object]) -> None:
    mark = "OK " if r["ok"] else "!! "
    print(f"{mark}{r['kind']}  {r['root']}")
    for k, v in (r.get("counts") or {}).items():
        print(f"     {k:<28} {v}")
    for p in (r.get("problems") or []):
        print(f"     PROBLEM: {p}")


def sort_dfdc_by_metadata(part_dir: os.PathLike, dest: os.PathLike,
                          move: bool = False) -> Dict[str, int]:
    """
    DFDC ships each part as a flat folder of .mp4 plus a metadata.json of
    {"abc.mp4": {"label": "FAKE"|"REAL", "original": ...}}. This turns that
    into the real/ + fake/ layout the manifest builder expects.
    """
    import json
    part_dir, dest = Path(part_dir), Path(dest)
    meta_path = part_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata.json in {part_dir}")
    meta = json.loads(meta_path.read_text())

    (dest / "real").mkdir(parents=True, exist_ok=True)
    (dest / "fake").mkdir(parents=True, exist_ok=True)
    counts = {"real": 0, "fake": 0, "missing": 0}
    op = shutil.move if move else shutil.copy2

    for fname, info in meta.items():
        src = part_dir / fname
        if not src.exists():
            counts["missing"] += 1
            continue
        label = "fake" if str(info.get("label", "")).upper() == "FAKE" else "real"
        op(str(src), str(dest / label / fname))
        counts[label] += 1

    log.info(f"sorted DFDC part: {counts}")
    return counts


# --------------------------------------------------------------------------
# Discovery: what do we actually have?
# --------------------------------------------------------------------------

def discover(cfg) -> Dict[str, Optional[Path]]:
    """
    Look at the roots configured in cfg.paths and report which datasets are
    usable. Returns {kind: root_or_None}.
    """
    found: Dict[str, Optional[Path]] = {}
    pairs = [("celebdf", cfg.paths.celebdf_root),
             ("ffpp", cfg.paths.ffpp_root),
             ("dfdcp", cfg.paths.dfdc_root),
             ("custom", cfg.paths.custom_root)]
    for kind, root in pairs:
        if root is None:
            found[kind] = None
            continue
        rep = verify(root, kind)
        found[kind] = Path(root) if rep["ok"] else None
    if not any(found.values()):
        print("\nNo usable dataset configured yet. That is expected on day one.")
        print("Run fsvnet.synthetic.build_synthetic_dataset(cfg) to exercise the")
        print("pipeline while the download requests are pending.")
    return found


__all__ = ["REGISTRY", "DatasetSpec", "print_acquisition_guide", "fetch_kaggle",
           "kaggle_available", "verify", "sort_dfdc_by_metadata", "discover",
           "VIDEO_EXTS"]
