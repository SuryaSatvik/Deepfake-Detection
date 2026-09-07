"""
test_qwen_path.py
=================

Tests everything in the Qwen detector path that does **not** need model weights,
so the parts that can be verified are verified before you spend two hours of GPU
time on the parts that cannot.

    python scripts/test_qwen_path.py

What is covered: preflight, evidence formatting, prompt structure, frame
selection, example construction, the ablation grid, and the answer-token
sanity rule.

What is NOT covered, and cannot be here: loading Qwen weights, LoRA attachment,
the actual forward pass, and training. Those need Hugging Face access. Run
`preflight()` on your own machine first — it is the substitute for this test on
that half of the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fsvnet.qwen_detector import (QWEN_ABLATIONS, QwenDetectorConfig,  # noqa: E402
                                  build_messages, format_evidence, preflight,
                                  select_frames)

results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 70)
    print("Qwen detector path — weight-free tests")
    print("=" * 70)

    cfg = QwenDetectorConfig()

    # ---- 1. preflight ------------------------------------------------
    print("\n1. Preflight")
    rep = preflight(cfg, verbose=True)
    check("preflight returns a report and never raises",
          isinstance(rep, dict) and "ok" in rep,
          f"ok={rep['ok']}, {len(rep['problems'])} blocker(s)")

    # ---- 2. evidence formatting --------------------------------------
    print("\n2. Evidence block")
    desc = {"temporal_whiteness": 0.412, "temporal_peakiness": 0.310,
            "flicker_index": -0.88, "spectral_drift": 0.004}
    pcts = {"temporal_whiteness": 3.0, "temporal_peakiness": 97.0,
            "flicker_index": 48.0, "spectral_drift": 2.0}
    txt = format_evidence(desc, pcts)
    print("\n" + txt + "\n")
    check("anomalous low percentile is flagged",
          "temporal_whiteness" in txt and "ANOMALOUS" in txt.split("\n")[1])
    check("anomalous high percentile is flagged",
          "ANOMALOUS" in [l for l in txt.split("\n") if "peakiness" in l][0])
    check("mid percentile is NOT flagged",
          "ANOMALOUS" not in [l for l in txt.split("\n") if "flicker_index" in l][0])
    check("context-only descriptor is never called anomalous",
          "ANOMALOUS" not in [l for l in txt.split("\n") if "spectral_drift" in l][0],
          "spectral_drift sits at the 2nd pct but is context, not evidence")
    check("uncalibrated values say so",
          "not calibrated" in format_evidence(desc, None))

    # ---- 3. prompt structure -----------------------------------------
    print("\n3. Prompt structure")
    m = build_messages(4, txt, answer=None)
    check("system + user turns at inference", len(m) == 2 and m[0]["role"] == "system")
    check("one image entry per frame",
          sum(1 for c in m[1]["content"] if c["type"] == "image") == 4)
    check("question is present", "real or manipulated" in m[1]["content"][-1]["text"])
    check("evidence is inside the user turn",
          "temporal_whiteness" in m[1]["content"][-1]["text"])

    mt = build_messages(4, txt, answer=" fake")
    check("training adds an assistant turn",
          len(mt) == 3 and mt[2]["role"] == "assistant")

    m0 = build_messages(0, txt, None)
    check("measurements-only mode has no images",
          sum(1 for c in m0[1]["content"] if c["type"] == "image") == 0)

    # ---- 4. frame selection ------------------------------------------
    print("\n4. Frame selection")
    check("evenly spaced and inclusive of both ends",
          select_frames(16, 4) == [0, 5, 10, 15], str(select_frames(16, 4)))
    check("fewer frames than requested returns all",
          select_frames(3, 4) == [0, 1, 2])
    check("deterministic", select_frames(16, 4) == select_frames(16, 4))

    # ---- 5. ablation grid --------------------------------------------
    print("\n5. Ablation grid")
    check("three variants defined", len(QWEN_ABLATIONS) == 3)
    check("frames-only is the Skyra-style baseline",
          QWEN_ABLATIONS["qwen_frames_only"] == {"use_frames": True,
                                                 "use_tsc_evidence": False})
    check("frames+TSC is the contribution",
          QWEN_ABLATIONS["qwen_frames_plus_tsc"]["use_tsc_evidence"] is True)
    check("the two differ in exactly one flag",
          sum(1 for k in QWEN_ABLATIONS["qwen_frames_only"]
              if QWEN_ABLATIONS["qwen_frames_only"][k]
              != QWEN_ABLATIONS["qwen_frames_plus_tsc"][k]) == 1,
          "so the difference between them is attributable")

    # ---- 6. example construction -------------------------------------
    print("\n6. Example construction from FSV-Net")
    try:
        import torch

        from fsvnet.config import Config
        from fsvnet.model import build_model
        from fsvnet.qwen_detector import build_examples

        c = Config(); c.device = "cpu"; c.data.num_frames = 4
        c.data.img_size = 112; c.model.pretrained = False
        fsv = build_model(c).eval()

        class _Loader:
            def __iter__(self):
                yield {"clip": torch.randn(2, 4, 3, 112, 112),
                       "mask": torch.ones(2, 4, dtype=torch.bool),
                       "label": torch.tensor([0, 1]),
                       "video_index": torch.tensor([0, 1])}

        ex = build_examples(fsv, _Loader(), c, reference=None)
        check("one example per clip", len(ex) == 2, f"{len(ex)}")
        check("frames are uint8 HWC",
              ex[0]["frames"].dtype == np.uint8 and ex[0]["frames"].shape[-1] == 3,
              str(ex[0]["frames"].shape))
        check("descriptors harvested from the TSC branch",
              len(ex[0]["descriptors"]) >= 6, str(sorted(ex[0]["descriptors"])[:3]))
        check("labels preserved", [e["label"] for e in ex] == [0, 1])
    except Exception as e:
        check("example construction", False, f"{e.__class__.__name__}: {e}")

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 70)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        for n, ok, d in results:
            if not ok:
                print(f"  FAILED: {n}  {d}")
    print("\nNOT tested here (needs Hugging Face access): weight loading, LoRA\n"
          "attachment, forward pass, training. Run preflight() on your machine\n"
          "before the first training job.")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
