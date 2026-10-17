"""
Known-answer tests for src/fsvnet/xai_eval.py. No checkpoint, no dataset, CPU, seconds.

    python scripts/test_xai_eval.py

Each stub model has a PLANTED answer — the decision depends on one grid unit,
one frame, one frequency band, or one branch — so every method is checked
against the truth rather than against another method.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fsvnet import xai_eval as X  # noqa: E402

H = W = 64; T = 6; G = 8
PLANT_UNIT = 2 * G + 5                    # row 2, col 5
PLANT_FRAME = 3
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


class PatchNet(nn.Module):
    """margin = 4 * sum_t c_t * mean(feat_t over the planted unit) - 1.
    feat = 1x1 conv (channel mean). c_t = 1 on PLANT_FRAME, 0.05 elsewhere."""
    def __init__(self):
        super().__init__()
        self.use = {"spatial": True}
        conv = nn.Conv2d(3, 1, 1, bias=False)
        conv.weight.data.fill_(1 / 3)
        self.spatial = nn.Module(); self.spatial.features = nn.Sequential(conv, nn.Identity())
        self.register_buffer("mask", X.unit_masks(H, W, G)[PLANT_UNIT])
        c = torch.full((T,), 0.05); c[PLANT_FRAME] = 1.0
        self.register_buffer("c", c)

    def forward(self, clip, mask=None, lmf=None, **kw):
        B = clip.shape[0]
        f = self.spatial.features(clip.reshape(B * T, 3, H, W)).reshape(B, T, H, W)
        s = (f * self.mask).sum((-2, -1)) / self.mask.sum()
        m = 4 * (s * self.c).sum(1) - 1
        return torch.stack([torch.zeros_like(m), m], 1), {}


def band(clip):
    """the evidence-carrying band, 0.2-0.3 x Nyquist"""
    return X.lowpass(clip, 0.3) - X.lowpass(clip, 0.2)


class BandNet(nn.Module):
    """margin = energy in the 0.2-0.3 Nyquist band - threshold. Low-passing at
    >= 0.3 keeps the band; the first grid cutoff that removes it is 0.2."""
    def __init__(self, thr):
        super().__init__(); self.use = {}; self.thr = thr

    def forward(self, clip, mask=None, lmf=None, **kw):
        m = (band(clip) ** 2).mean((1, 2, 3, 4)) - self.thr
        return torch.stack([torch.zeros_like(m), m * 50], 1), {}


class BranchNet(nn.Module):
    """Linear head over three branch embeddings -> Shapley is known in closed form."""
    def __init__(self):
        super().__init__()
        self.branch_names = ["a", "b", "c"]
        self.fusion = lambda *e: (torch.cat(e, 1), None)
        self.classifier = nn.Linear(6, 2)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[0, 0, 0, 0, 0, 0],
                                                      [3., 1, -2, 0, 0.5, .5]]))
            self.classifier.bias.zero_()


def planted_clip(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, 3, H, W, generator=g) * 0.3
    x[:, PLANT_FRAME] += X.unit_masks(H, W, G)[PLANT_UNIT] * 2.0
    return x


def main():
    torch.manual_seed(0)
    masks = X.unit_masks(H, W, G)
    model = PatchNet().eval()
    clip = planted_clip()
    sc = X.ClipScorer(model, clip, None, batch=16)
    print("\nscorer and units")
    check("predicted fake on planted clip", sc.sign == 1 and sc.p0 > 0.8, f"p0={sc.p0:.3f}")
    check("unit masks partition the image", torch.allclose(masks.sum(0), torch.ones(H, W)))
    mean = X.make_fill("mean", clip, torch.zeros(1, 1, 3, H, W))
    roll = X.make_fill("roll", clip)
    check("roll fill = frame t+T/2", torch.equal(roll[:, 0], clip[:, T // 2]))

    print("\nattribution finds the planted unit")
    gc = X.grad_cam_units(model, clip, None, sc.sign, masks)
    check("Grad-CAM top unit", int(np.argmax(gc)) == PLANT_UNIT)
    ig, resid = X.integrated_gradients_units(model, clip, None, sc.sign, mean, masks, steps=16)
    check("IG top unit", int(np.argmax(ig)) == PLANT_UNIT)
    check("IG completeness residual small (linear model)", resid < 1e-3, f"{resid:.2e}")
    oc = X.occlusion_units(sc, mean, masks)
    check("occlusion top unit", int(np.argmax(oc)) == PLANT_UNIT)
    ks = X.kernel_shap_units(sc, mean, masks, n_samples=256, seed=1)
    check("KernelSHAP top unit", int(np.argmax(ks)) == PLANT_UNIT)
    gap = sc.margin0 - float(sc.margins(mean)[0])
    check("KernelSHAP efficiency (sum phi = f(x) - f(empty))", abs(ks.sum() - gap) < 1e-2,
          f"{ks.sum():.3f} vs {gap:.3f}")

    print("\nfaithfulness: a real ranking beats random, random does not")
    rng = np.random.default_rng(0)
    good = X.summarise_curves(X.curves(sc, mean, masks, X.ranking(oc)), sc.p0)
    rnd = [X.summarise_curves(X.curves(sc, mean, masks, rng.permutation(64)), sc.p0)
           for _ in range(5)]
    rdel = np.mean([r["del_auc"] for r in rnd])
    check("deletion AUC: method < random", good["del_auc"] < rdel - 0.2,
          f"{good['del_auc']:.3f} vs {rdel:.3f}")
    check("insertion AUC: method > random", good["ins_auc"] > np.mean([r["ins_auc"] for r in rnd]) + 0.2)
    check("flips after deleting 1 unit", good["flip_frac"] <= 0.01 + 1e-9, f"{good['flip_frac']}")
    check("comprehensiveness@5% large", good["comp@0.05"] > 0.4)
    check("|sufficiency| small once the unit is kept", good["suff@0.05"] < 0.05)
    d = X.paired_diff([r["del_auc"] for r in rnd[:3]], [r["del_auc"] for r in rnd[2:]])
    check("random vs random: not resolved", d["lo"] <= 0 <= d["hi"], f"p={d['p']:.2f}")
    d2 = X.paired_diff(np.full(30, rdel),
                       np.full(30, good["del_auc"]) + np.random.default_rng(1).normal(0, .01, 30))
    check("paired_diff resolves a real gap", d2["lo"] > 0 and d2["p"] < 0.01)
    dw = X.paired_diff([1, 1, 1, 0, 0, 0.], [0] * 6, w=[1, 1, 1, 3, 3, 3])
    check("paired_diff honours weights", abs(dw["mean"] - 0.25) < 1e-9, f"{dw['mean']}")

    print("\nstability and sanity")
    oc2 = X.occlusion_units(X.ClipScorer(model, X.add_pixel_noise(clip, 0.01), None, 16), mean, masks)
    check("occlusion stable under sigma=0.01", X.spearman(oc, oc2) > 0.9 and X.topk_jaccard(oc, oc2) > 0.9)
    check("constant maps: Jaccard is NaN, not a perfect 1.0",
          np.isnan(X.topk_jaccard(np.zeros(64), np.zeros(64))) and X.is_degenerate(np.zeros(64)))
    check("centre prior differs from the planted answer",
          int(np.argmax(X.centre_prior_units(G))) != PLANT_UNIT)

    print("\ncounterfactuals")
    tsc = X.temporal_scores(sc)
    check("temporal occlusion top frame", int(np.argmax(tsc)) == PLANT_FRAME)
    k = X.temporal_counterfactual(sc, X.ranking(tsc))
    check("freezing 1 frame flips", k == 1, f"k={k}")
    hf = torch.randn(1, T, 3, H, W) * 0.5
    sc2 = X.ClipScorer(BandNet(thr=float((band(hf) ** 2).mean()) * 0.5), hf, None, 16)
    cf = X.spectral_counterfactual(sc2)
    check("spectral CF: first flip at 0.2 Nyquist, monotone below",
          cf["cutoff_nyq"] == 0.2 and cf["monotone"], f"{cf['cutoff_nyq']}")
    check("lowpass(1.0) is identity", torch.allclose(X.lowpass(hf, 1.0), hf, atol=1e-5))
    check("lowpass keeps shape", X.lowpass(hf, 0.3).shape == hf.shape)

    print("\nexact branch Shapley")
    e = {"a": torch.tensor([[1., 2]]), "b": torch.tensor([[1., 0]]), "c": torch.tensor([[2., 2]])}
    sh = X.branch_shapley(BranchNet(), e)
    check("linear head -> phi = w.e", np.allclose(sh["phi"][0], [5, -2, 2]), f"{sh['phi'][0]}")
    check("efficiency residual ~ 0", sh["efficiency_residual"][0] < 1e-5)

    print("\nBorda")
    b = X.borda({"m": {"d": .1, "i": .9}, "r": {"d": .5, "i": .5}}, {"d": "lower", "i": "higher"})
    check("better method gets more points", b["m"] > b["r"])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL); sys.exit(1)


if __name__ == "__main__":
    main()
