# Pre-training findings — Celeb-DF-v2, 27 Aug 2026

Everything in this file is **measured on the data in this repository**. Nothing
is estimated, quoted from a paper, or carried over from an earlier write-up.
Reproduce any of it with the script named beside it.

---

## 1. What the dataset actually is

`scripts/build_celebdf_manifest.py` -> `data/manifests/`

The `Celeb-DF-v2/` folder is **not** the raw video distribution. It is a
pre-extracted derivative:

| property | value |
|---|---|
| clips | 6,526 (587 Celeb-real, 300 YouTube-real, 5,639 Celeb-synthesis) |
| format | PNG face crops, **256x256 RGB**, ~32 per clip |
| temporal grid | ~32 frames dumped per video, so the stride between filenames is **not constant** (measured 5-50); filenames are true source indices, so `index/fps` is still a valid timestamp |
| landmarks | **81 points per frame**, `.npy` uint32, one dir per clip |
| total frames | 207,948 |
| class balance | 887 real / 5,639 fake — **86.4% fake** |
| identity groups | 304 (union-find over every identity pair a swap touches) |
| approx duration | 12.6 s mean (2.9 – 24.6 s), fps assumed 30 |

Four consequences that constrain every design decision:

1. **No audio, no codec metadata, no arbitrary resampling.** Any claim about
   audio deepfake detection would be fabricated — the Streamlit app must not
   offer it.
2. **Face detection is a no-op.** `video_io.py` / `faces.py` are bypassed on
   this dataset; `celebdf_frames.py` reads the PNGs directly.
3. **All timestamps are reconstructed** as `source_index / 30`. They are
   labelled `approx` throughout and `assumed_fps` is carried in the manifest.
4. **The landmarks are the one resource nothing else in this project uses.**
   That is what makes `lmf.py` possible and is the reason the proposed
   architecture looks the way it does.

---

## 2. The split, and why the official one cannot be used

`scripts/build_celebdf_manifest.py`

**Identity-disjoint split (the one used everywhere):** train 4,574 / val 976 /
test 976, zero identity groups appearing in more than one split.

**Leakage audit — a random split over clips would put 95.1% of test clips in
the same identity group as training data** (932 of 980). That is the number a
random-split AUC would be measuring. It is not a small correction.

**The official Celeb-DF-v2 test list is itself not identity-disjoint.** Its
517 clips touch 74 identity groups, and those groups collectively contain
**6,296 clips — 96.5% of the entire dataset**. Forcing whole identity groups
into test leaves 96 training clips, all real. So:

> Published Celeb-DF-v2 numbers computed on the official protocol are measured
> with training identities present in the test set. This project reports the
> identity-disjoint number as its headline and can report the official-protocol
> number only as a separately labelled, identity-leaky comparison.

---

## 3. Do the hand-crafted forensic descriptors survive across identities?

`scripts/extract_lmf.py`, `scripts/extract_tsc.py`, `scripts/compare_branch_probes.py`
-> `results/comparison/branch_probe_comparison.json`

2,087 clips (all 887 reals + 400 fakes per split), 8 frames each, identical
clips and identical split for both channels. Classifiers are a linear probe and
a gradient-boosted tree on the closed-form descriptors only.

| channel | model | test AUC | 95% CI | PR-AUC | Bal. Acc |
|---|---|---:|---|---:|---:|
| LMF only | linear | 0.5134 | 0.465 – 0.559 | 0.615 | 0.503 |
| LMF only | gbdt | 0.5407 | 0.493 – 0.586 | 0.655 | 0.539 |
| **TSC only** | **linear** | **0.5867** | **0.541 – 0.630** | 0.692 | 0.563 |
| TSC only | gbdt | 0.5401 | 0.494 – 0.584 | 0.650 | 0.519 |
| LMF + TSC | linear | 0.5421 | 0.492 – 0.589 | 0.628 | 0.517 |
| LMF + TSC | gbdt | 0.5475 | 0.499 – 0.595 | 0.657 | 0.508 |

Paired bootstrap on the difference (the correct test — the models are scored on
the same clips, so their separate CIs are not independent):

| comparison | delta AUC | 95% CI | verdict |
|---|---:|---|---|
| TSC − LMF | −0.0000 | −0.063 – +0.062 | **not resolved** |
| (LMF+TSC) − LMF | +0.0072 | −0.027 – +0.041 | **not resolved** |
| (LMF+TSC) − TSC | +0.0072 | −0.047 – +0.060 | **not resolved** |

### Read this honestly

**Every closed-form channel sits between 0.51 and 0.59 AUC on unseen
identities.** Only TSC-linear has a confidence interval that excludes 0.5, and
only just. No pairwise difference is resolved. On this evidence, neither
hand-crafted forensic summary carries a strong identity-generalisable
manipulation signal on Celeb-DF-v2.

### The leakage demonstration

`scripts/lmf_leakage_demo.py` -> `results/baseline/lmf_leakage_demo.json`

The same LMF features, the same classifier, different split regimes:

| split regime | linear AUC | gbdt AUC |
|---|---:|---:|
| identity-disjoint (honest) | 0.5134 | 0.5407 |
| random split over clips | 0.6147 | 0.5987 |
| random split, Celeb-only | 0.6093 | 0.6000 |

**~0.10 AUC of apparent performance is identity leakage and nothing else.**

An earlier same-identity sample (30 real vs 30 fake, all `id0`–`id2`) showed
Cohen's *d* of −1.18 on `inner_outer_contrast` and +1.05 on `boundary_excess`.
Those effect sizes did not survive the identity-disjoint split. They were
measuring *who the person is*, not *whether the clip is fake*. This is exactly
the failure the image stage of this capstone hit (0.7998 random-split AUC vs
0.5702 held-out), reproduced at video level with a controlled measurement.

Two further controls:
* Descriptors do **not** directly predict identity (CV accuracy 0.654 vs 0.658
  chance on the 4 largest Celeb-real groups), so the leak is not simple
  identity encoding — it is capture-condition confounding.
* Among **real clips only**, the descriptors separate YouTube-real from
  Celeb-real at **AUC 0.639**. There is a source/capture confound in the data,
  and any model is free to exploit it unless the split prevents it.

### What this does and does not license

It does **not** show the branches are worthless. Both probes compress a whole
clip to 7–10 scalars; the learned branches consume the full `(T-1, 3)` region
energy series and the `(F, K)` TSC map alongside a spatial CNN. These numbers
are a **lower bound**.

It does mean the project must **not** be built on the assumption that either
forensic channel works. The contribution has to be demonstrated by the
ablation, and a negative ablation is a real result to report, not a gap to
fill.

---

## 6. The year-one image stage, audited

`scripts/image_stage_leakage_audit.py` -> `results/baseline/image_stage_leakage.json`

### Correction to an earlier claim in this repository

An earlier README asserted that the image stage scored **0.7998 AUC on a random
split and 0.5702 held out**, with **17.7%** of test rows having a near-duplicate
in train. Those figures appear in **no notebook, no CSV and no output file** in
this project. They are unsubstantiated and must not be presented. What follows
replaces them with measurements.

### What year one actually produced

Eight executed notebooks and five trained checkpoints, with these recorded results:

| notebook | result |
|---|---|
| `balanced_training_clean.ipynb` | best val accuracy **0.9628** (EfficientNet-B0, balanced) |
| `model_evaluation_and_explainability.ipynb` | test accuracy **0.9630** over 3,165 images |
| `proposed_model_training.ipynb` | best val **0.9527**, test accuracy **0.9536** |
| `model_comparison.ipynb` | baseline **0.9823** vs proposed **0.9536** (post-normalisation) |
| `fsw_novel_detector.ipynb` | FSW-Net, 15 epochs, best val **0.9324**, test accuracy **0.9302** |

Dataset: 21,090 images (10,538 deepfake, 10,140 real, 250 edited, 157
ai-generated), split 14,757 / 3,168 / 3,165.

### The measured contamination of that test set

Filenames carry no identity or source-video token, so identity grouping is
impossible after the fact. A 64-bit dHash nearest-neighbour search against the
training set is the only handle left — which is itself the lesson: **record
identity at collection time or you cannot audit later.**

| Hamming threshold | meaning | test images with a train match |
|---:|---|---:|
| 0 | byte-identical hash | 72 (**2.3%**) |
| 2 | visually identical | 82 (2.6%) |
| 5 | near-duplicate | 219 (**6.9%**) |
| 10 | same scene / shot | 1,532 (**48.4%**) |

By class at threshold 5: real 7.5%,
deepfake 6.5%,
ai_generated 4.2%,
edited 0.0%.

**Report the sweep, not one threshold.** The honest statement is: *6.9% of the
year-one test set are near-duplicates of training images, and 48.4% share a
scene with training data.* The 96.3% test accuracy was computed over that set,
so an unknown but non-trivial share of it measured recall rather than
detection. The strict figure (6.9%) is the defensible headline; the loose one
(48.4%) is the one that explains why the video stage was rebuilt around
identity-disjoint splitting from the start.

---

## 4. Skyra — what it actually is

Sources: [arXiv 2512.15693](https://arxiv.org/abs/2512.15693),
[github.com/JoeLeelyf/Skyra](https://github.com/JoeLeelyf/Skyra/)

Corrections to the previous README in this repository:

| previous claim | actual |
|---|---|
| "no released weights" | **weights are released** on HuggingFace; CVPR 2026 |
| implied comparability on Celeb-DF | **Skyra is not a face-swap detector** |

* Base: **Qwen2.5-VL-7B-Instruct**.
* Stage 1: SFT on **ViF-CoT-4K** (~5K videos, 3.5K real from Panda-70M, 1.5K
  from Kinetics-400) with fine-grained human artifact annotations + CoT.
* Stage 2: **GRPO** with an asymmetric accuracy reward (false positives
  penalised −0.2) plus an inspection/format reward.
* Three-layer artifact taxonomy: *low-level forgery* and *violation of laws* ->
  8 categories (colour/light, texture, motion forgery, object inconsistency,
  interaction inconsistency, causality, commonsense, unnatural movement) ->
  fine-grained observables.
* Trained and evaluated on **fully generated video** — Wan2.1/2.2, CogVideoX,
  Hunyuan, Sora-2, LTX, MiniMax-Hailuo.
* **It is not evaluated on Celeb-DF or FaceForensics++ at all.**

### Why this reframes the whole project

Running Skyra on Celeb-DF-v2 is an **out-of-distribution transfer test**, not a
like-for-like baseline. Its taxonomy is about physics, causality and object
persistence; a Celeb-DF face swap violates none of those — it is a blended
region inside an otherwise real, physically consistent video.

So the honest research question is **not** "can we beat Skyra". It is:

> A state-of-the-art grounded-artifact-reasoning detector for generated video
> has a taxonomy with no category for localised face replacement. How far does
> it transfer to face-swap forensics, and does conditioning it on
> face-geometry-derived forensic evidence close the gap?

That is a better contribution than a leaderboard delta, and it is one this
hardware can actually support. Any table claiming "we beat Skyra" without this
framing would be misleading and should not be written.

---

## 5. Hardware reality

| environment | CPU | RAM | GPU |
|---|---|---|---|
| Cowork VM (mounted folders) | 2 | 3 GB | none |
| cloud container | 2 | 7 GB | none |
| Surya's machine | — | — | RTX 4060, 8 GB |

Neither environment reachable from this session has CUDA. Everything above was
computed on CPU. Skyra is a 7B model: on an 8 GB 4060 it is **NF4 inference
only** — fine-tuning it is not possible on that card, so the Skyra condition is
zero-shot/prompted, and any fine-tuned proposed model must use a smaller
backbone (Qwen2.5-VL-3B) for the comparison to be honest about what changed.
