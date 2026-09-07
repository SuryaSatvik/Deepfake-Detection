"""
Explainability evidence for the report, on real test-split clips.

    python scripts/eval_conditions.py          # run this FIRST
    python scripts/make_explainability.py

Picks four identity-disjoint TEST clips using the proposed model's own scores
(from results/scores_C3.npz, so the selection is reproducible and not by eye):
a confidently-correct fake, a confidently-correct real, and the model's worst
false positive and worst false negative. For each it runs the whole
explain_clip() stack -- Grad-CAM over the spatial branch, temporal occlusion in
freeze mode, branch attribution by zeroing, TSC map and descriptors -- plus the
LMF measurement pipeline, and dumps every array to
results/explain/<tag>.npz + a summary results/explain/summary.json.

Nothing is plotted here. Arrays are dumped so the report figures can be drawn
in one consistent style without re-running the GPU.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsvnet import Config, celebdf_frames as cf, lmf as LMF
from fsvnet.model import build_model
from fsvnet.dataset_frames import normalise
from fsvnet.explain import explain_clip

NUM_FRAMES, IMAGE_SIZE, SEED = 16, 224, 42
MAN = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
TPL = ROOT / "data" / "manifests" / "lmf_template.npy"
CKPT = ROOT / "models" / "checkpoints" / "fsvnet_dfe_C3.pt"
OUT = ROOT / "results" / "explain"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pick(df, scores):
    """Four cases chosen by the model's own test scores, not by eye."""
    te = df[df.split == "test"].reset_index(drop=True)
    y, p = scores["y_test"], scores["p_test"]
    if len(te) != len(y):
        raise SystemExit(f"scores_C3.npz has {len(y)} rows, test split has {len(te)}. "
                         "Re-run scripts/eval_conditions.py -- the loader order must match.")
    fake, real = np.where(y == 1)[0], np.where(y == 0)[0]
    return [("true_fake", int(fake[np.argmax(p[fake])]), "most confident correct fake"),
            ("true_real", int(real[np.argmin(p[real])]), "most confident correct real"),
            ("false_positive", int(real[np.argmax(p[real])]), "worst false positive"),
            ("false_negative", int(fake[np.argmin(p[fake])]), "worst false negative")]


def main():
    print(f"device: {DEVICE}")
    if not CKPT.exists():
        raise SystemExit(f"missing {CKPT} -- train condition C3 first.")
    sc = ROOT / "results" / "scores_C3.npz"
    if not sc.exists():
        raise SystemExit("missing results/scores_C3.npz -- run scripts/eval_conditions.py first.")

    torch.manual_seed(SEED); np.random.seed(SEED)
    df = cf.load_manifest(MAN, ROOT)
    scores = np.load(sc)
    template = np.load(TPL)
    masks = LMF.region_masks(template, 256)

    blob = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    c = Config()
    c.data.num_frames, c.data.img_size, c.data.seed = NUM_FRAMES, IMAGE_SIZE, SEED
    c.model.use_tsc, c.model.use_lmf = blob["use_tsc"], blob["use_lmf"]
    m = build_model(c).to(DEVICE); m.load_state_dict(blob["state_dict"]); m.eval()

    te = df[df.split == "test"].reset_index(drop=True)
    summary = {"checkpoint": CKPT.name, "device": DEVICE, "cases": []}

    for tag, idx, why in pick(df, scores):
        rec = te.iloc[idx]
        print(f"\n=== {tag}: {rec.clip_name}  truth={rec.label_name}  "
              f"p(fake)={scores['p_test'][idx]:.4f}  ({why})")

        # --- model-side explanation, at the training resolution
        fr224, _, meta = cf.load_clip(rec.video_path, rec.landmark_path,
                                      n_frames=NUM_FRAMES, img_size=IMAGE_SIZE)
        clip = normalise(fr224).unsqueeze(0).to(DEVICE)
        ex = explain_clip(m, clip, reference=None, do_cam=True, do_occlusion=True)

        # --- LMF measurement, at the 256 px the descriptors were defined on
        fr256, lm256, meta256 = cf.load_clip(rec.video_path, rec.landmark_path,
                                             n_frames=8, img_size=256)
        r = LMF.analyse_clip(fr256, lm256, template, size=256, keep_maps=True)

        np.savez_compressed(
            OUT / f"{tag}.npz",
            frames=ex.frames,
            cam=ex.cam if ex.cam is not None else np.zeros(0),
            occl_delta=ex.occlusion["delta"] if ex.occlusion else np.zeros(0),
            occl_baseline=np.array([ex.occlusion["baseline"]]) if ex.occlusion else np.zeros(0),
            tsc_map=ex.tsc_map if ex.tsc_map is not None else np.zeros(0),
            frame_attention=ex.frame_attention if ex.frame_attention is not None else np.zeros(0),
            timestamps=np.asarray(meta["timestamp_s_approx"], dtype=float),
            lmf_frames=fr256, lmf_warped=r.warped, lmf_residual=r.residual,
            lmf_energy=r.energy, lmf_fit_err=r.fit_err,
            lmf_timestamps=np.asarray(meta256["timestamp_s_approx"], dtype=float),
            mask_inner=masks["inner"], mask_boundary=masks["boundary"], mask_outer=masks["outer"],
        )

        summary["cases"].append({
            "tag": tag, "why": why, "clip": str(rec.clip_name),
            "truth": str(rec.label_name), "p_fake_eval": float(scores["p_test"][idx]),
            "p_fake_here": ex.fake_probability, "verdict": ex.verdict,
            "attribution": ex.attribution,
            "descriptors": ex.descriptors,
            "lmf_descriptors": {k: float(v) for k, v in r.descriptors.items()},
            "top_frames": ex.top_frames(3),
            "summary_lines": ex.summary_lines(),
        })
        for line in ex.summary_lines():
            print("   " + line)

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/*.npz and summary.json")


if __name__ == "__main__":
    main()
