"""
Fetch the trained checkpoints from the GitHub release into models/checkpoints/.

    python scripts/download_weights.py            # all four conditions
    python scripts/download_weights.py --only C3  # just the proposed model

The weights are not in git — four checkpoints at ~37 MB each would bloat every
clone for the many people who only want to read the code. They are attached to
a release instead, which is what releases are for.

Nothing here is required to run the app: with no checkpoint present the
Streamlit frontend runs in evidence-only mode and says so.
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "checkpoints"

REPO = "SuryaSatvik/Deepfake-Detection"
TAG = "v1.0-weights"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

CONDITIONS = {
    "B0": ("fsvnet_dfe_B0.pt", "conventional — spatial + frequency + SRM"),
    "C1": ("fsvnet_dfe_C1.pt", "+ TSC"),
    "C2": ("fsvnet_dfe_C2.pt", "+ LMF  (needs 81-point landmarks at inference)"),
    "C3": ("fsvnet_dfe_C3.pt", "+ TSC + LMF  — the proposed model"),
}


def human(n):
    return f"{n / 1e6:.1f} MB"


def fetch(name: str) -> bool:
    out = DEST / name
    if out.exists():
        print(f"  {name:<22} already present ({human(out.stat().st_size)}); skipping")
        return True
    url = f"{BASE}/{name}"
    tmp = out.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            h = hashlib.sha256()
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r  {name:<22} {pct:5.1f}%  {human(done)}",
                              end="", flush=True)
        tmp.replace(out)
        print(f"\r  {name:<22} done   {human(out.stat().st_size)}  "
              f"sha256 {h.hexdigest()[:12]}")
        return True
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        print(f"\r  {name:<22} FAILED  HTTP {e.code}")
        if e.code == 404:
            print(f"      Not found at {url}\n"
                  f"      The release may not be published yet — check "
                  f"https://github.com/{REPO}/releases")
        return False
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        print(f"\r  {name:<22} FAILED  {e.__class__.__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="COND", choices=list(CONDITIONS),
                    help="conditions to fetch (default: all)")
    a = ap.parse_args()

    want = a.only or list(CONDITIONS)
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"destination: {DEST}\nsource:      {BASE}\n")

    ok = True
    for cid in want:
        name, desc = CONDITIONS[cid]
        print(f"[{cid}] {desc}")
        ok &= fetch(name)

    print()
    have = sorted(p.name for p in DEST.glob("fsvnet_dfe_*.pt"))
    print(f"checkpoints present: {len(have)}  {have}")
    if not ok:
        print("\nSome downloads failed. The app still runs without them, in "
              "evidence-only mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
