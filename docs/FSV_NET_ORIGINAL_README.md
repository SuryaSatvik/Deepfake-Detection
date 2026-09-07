# FSV-Net — Frequency-Spatial-Watermark **Video** Network

Video deepfake detection built on the image-level FSW-Net in `../DeepfakeProject`,
with two contributions and one methodological correction.

---

## What is new here

### 1. Temporal Spectral Coherence (TSC) — `src/fsvnet/tsc.py`

Every frequency-domain detector so far asks *does this frame's spectrum look
synthetic?* TSC asks **does the spectrum evolve over time the way a camera's does?**

In real footage the *fine* spatial detail is dominated by photon and read noise, which
is close to independent frame to frame — so the energy in the fine radial rings
fluctuates like **white noise along the time axis**. In a synthesised or swapped video
that detail is *computed* from a temporally smooth latent, so it becomes temporally
**coloured** (over-smooth), or, where blending and colour correction are involved,
**spiked** at one flicker rate.

For a clip of `T` frames:

1. per frame, the azimuthally-averaged power spectrum `A ∈ R^{T×K}`
2. log-compress, subtract the temporal mean and linear drift per ring
3. Hann-window, then rFFT **along time** → `C ∈ R^{F×K}`, the coherence map
4. a small CNN embeds `C`; closed-form scalars (`temporal_whiteness`,
   `temporal_peakiness`, signed `flicker_index`, …) go to the report and the
   evidence pack

**129,204 parameters — 1.4% of the model.** This mirrors the LFWS/LFWL result
(May 2026), where a 292-parameter handcrafted fusion block moved FF++ from 74.8 to
78.6 AUC: cheap forensic cues bolted onto a learned backbone buy real robustness.

### 2. Evidence-conditioned reasoning — `src/fsvnet/evidence.py`, `vlm_reasoner.py`

Skyra-style grounded artefact reasoning, with one change: the VLM is conditioned on the
detector's **percentile-calibrated measurements**, not only on frames. It is told
*"temporal whiteness 0.41 — the 3rd percentile of real videos"*, not a bare float. A
VLM given only a probability always invents a reason for it; given calibrated
measurements it can say "the spatial evidence is unremarkable, only the temporal
statistic is anomalous", which is both more useful and falsifiable.

> **Naming, honestly:** Skyra (arXiv 2512.15693) has no released weights — it is a
> Qwen2.5-VL-7B fine-tuned with GRPO on a human-annotated artefact dataset. This is
> Skyra-*style* reasoning on **base** Qwen2.5-VL. Describe it that way in the report.

### 3. Identity-disjoint splitting — `src/fsvnet/manifest.py`

The image stage of this capstone reported 0.7998 [UNVERIFIED - see docs/FINDINGS.md S6] AUC on a random split and **0.5702 [UNVERIFIED - see docs/FINDINGS.md S6]**
on a genuinely held-out one, because 17.7% [UNVERIFIED - see docs/FINDINGS.md S6] of its test rows had a near-duplicate in
train. Video data leaks worse: one video is many near-identical frames, one person
appears in dozens of videos, and a swap `id0 → id16` binds two identities together.

So the manifest builds identity groups with a **union-find over every identity pair a
manipulation touches** and splits over the connected components. `leakage_audit()`
measures what a random split *would* have leaked, so the report can quote it.

---

## Layout

```
video_detector/
├── notebooks/
│   ├── FSV_Net_RUN_ALL.ipynb   ← THE notebook. One file, run top to bottom.
│   │                             Ships with its outputs already in it.
│   └── FSV_Net_main.ipynb      (superseded — a pointer to the one above)
├── src/fsvnet/
│   ├── config.py               all tunables + hardware auto-tuning
│   ├── utils.py                seeding, logging, device, OOM guard
│   ├── datasets_fetch.py       acquisition guide, Kaggle fetch, layout checks
│   ├── synthetic.py            controlled synthetic clips + the TSC unit test
│   ├── video_io.py             decoding, uniform temporal sampling
│   ├── faces.py                detection + temporally smoothed crop trajectory
│   ├── manifest.py             identity-disjoint splits, leakage audit
│   ├── dataset.py              PNG filmstrip cache, augmentation, Dataset
│   ├── tsc.py                  ★ Temporal Spectral Coherence
│   ├── branches.py             spatial / frequency / SRM + temporal encoder
│   ├── model.py                FSV-Net assembly, FSW-Net warm start
│   ├── losses.py               focal loss + branch-diversity regulariser
│   ├── train.py                AMP, grad accumulation, video-level selection
│   ├── evaluate.py             video-level metrics, ablations, figures
│   ├── evidence.py             ★ the forensic evidence pack
│   ├── vlm_reasoner.py         Qwen2.5-VL / Claude / template, with fallbacks
│   ├── inference.py            single-video end-to-end pipeline
│   ├── explain.py              ★ Grad-CAM / occlusion / branch attribution
│   ├── qwen_detector.py        ★ Qwen2.5-VL as the detector, QLoRA
│   ├── benchmark.py            ★ capability table + fair-comparison harness
│   └── viz.py                  one palette + type scale for every figure
├── app/
│   └── streamlit_app.py        ★ the frontend
├── scripts/
│   ├── prepare_data.py         CLI: guide / verify / synthetic / manifest / preprocess
│   ├── smoke_test.py           21 assertions, no dataset, no GPU, ~2 min
│   └── test_qwen_path.py       23 assertions on the Qwen path, no weights needed
├── configs/ checkpoints/ data/ outputs/
└── requirements.txt
```

---

## Quick start

```bash
cd video_detector
python -m venv .venv && .venv\Scripts\Activate.ps1        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. prove the pipeline works — no dataset, no GPU, ~2 minutes
python scripts/smoke_test.py

# 2. find out what to download and start the gated requests TODAY
python scripts/prepare_data.py guide

# 3. everything else — one notebook, run all cells
jupyter notebook notebooks/FSV_Net_RUN_ALL.ipynb

# 4. the frontend (needs a checkpoint in checkpoints/ — step 3 makes one)
streamlit run app/streamlit_app.py
```

`FSV_Net_RUN_ALL.ipynb` is the only notebook you need. It generates its own dataset,
trains, evaluates, ablates and explains, with `DEMO_MODE = True`; flip that to `False`
and point it at Celeb-DF to run for real. It is delivered **already executed**, so you
can read every figure and number before running anything.

`smoke_test.py` asserts 21 things, including the one that matters: on
variance-matched synthetic clips whose *average spatial spectra are identical*, TSC
scores AUC 1.00 while a per-frame spectral detector stays at chance. If that ever
fails, the novelty claim is broken — not the code style.

---

## Hardware

`Config.auto_tune()` inspects the card and rewrites clip length, batch size, AMP and
VLM size to fit. On the target **RTX 4060 8 GB** it selects 16 frames, batch 2 with
8-step accumulation (effective 16), AMP on, and Qwen2.5-VL-3B in NF4.

| VRAM | frames | batch × accum | VLM |
|---|---|---|---|
| CPU only | 8 | 1 × 4 | none (template) |
| < 6 GB | 8 | 1 × 16 + checkpointing | 3B NF4 |
| **8 GB (4060)** | **16** | **2 × 8** | **3B NF4** |
| 12–16 GB | 24 | 4 × 4 | 7B NF4 |
| 24 GB+ | 32 | 8 × 2 | 7B bf16 |

If you still hit CUDA OOM, in this order: `grad_checkpointing = True` →
`num_frames = 8` → `batch_size = 1` → disable the spatial branch (a legitimate
ablation that uses almost no memory).

**The VLM and training do not coexist on 8 GB.** Call `free_cuda()` before loading the
reasoner; the notebook does this for you.

---

## Datasets

Nothing here can download them for you — all of them are gated, and approval takes
1–3 days. **Start the requests before you write another line of code.**

| Dataset | Access | Size | Role |
|---|---|---|---|
| Celeb-DF v2 | request form | ~9 GB | primary train/val/test |
| FaceForensics++ c23 | signed EULA | ~40 GB (`-n 200` for a subset) | secondary + cross-manipulation |
| DFDC Preview | Kaggle rules | ~5 GB | held-out cross-dataset test only |
| synthetic | none | ~200 MB | pipeline smoke test + TSC unit test |

`python scripts/prepare_data.py guide` prints the links and the exact commands.
`verify` checks a download's layout before you spend an hour preprocessing it.

---

## The results you need

| # | Table / figure | Produced by |
|---|---|---|
| 1 | Controlled TSC experiment — matched marginals, TSC ≈ 1.0, control ≈ 0.5 | notebook §3 |
| 2 | Leakage audit — % of a random split's test videos sharing an identity with train | §4 |
| 3 | Ablation: `spatial_freq_srm` vs `full` (± TSC), with bootstrap CIs | §9 |
| 4 | Cross-dataset transfer (train Celeb-DF → test DFDC-P) | §14 |
| 5 | TSC maps, real vs fake, side by side | §3 |
| 6 | Groundedness of explanations, with and without calibration | §11 |

### What the shipped demo run actually produced

On 120 synthetic clips (20 subjects × 3 clips, `smooth` fakes only), CPU, ~13 minutes:

| variant | test AUC | 95% CI |
|---|---|---|
| spatial only | 0.358 | 0.156 – 0.581 |
| spatial + freq + SRM (conventional) | 0.827 | 0.592 – 0.988 |
| **TSC only** | **1.000** | 1.000 – 1.000 |
| **full (+ TSC)** | **1.000** | 1.000 – 1.000 |

Controlled experiment: TSC AUC **1.000**, per-frame spectral controls **0.455** and
**0.536**. Leakage audit: a random split would have put **100%** of test videos in the
same identity group as training data; the identity-disjoint split puts **0%**.

**Read those numbers for what they are.** The synthetic fakes were built with exactly
the defect TSC looks for, so this is a sanity check that the machinery works, not
evidence about real deepfakes. The number that matters is still the one you get from
Celeb-DF.

Report **video-level** numbers only. Frame- and clip-level metrics are systematically
optimistic because clips from one video are not independent samples.

---

## Honest limitations

* **TSC is a hypothesis with a mechanism, not a proven result.** It is validated on
  synthetic clips where the signal was planted deliberately. Whether it survives H.264
  rate control on real footage is what notebook §3 → §9 answers, and a negative answer
  is a publishable finding too.
* **H.264 rate control also colours the temporal spectrum.** This is why the JPEG /
  rescale augmentation in `dataset.py` is mandatory rather than optional — without it
  the model can learn the codec instead of the generator.
* **A static scene has little fine detail to measure**, so TSC is weakest exactly where
  there is least motion.
* **The FSW-Net warm start transfers features, not accuracy.** The image model's honest
  held-out AUC was 0.5702 [UNVERIFIED - see docs/FINDINGS.md S6]; nothing about its performance carries over.
* **The groundedness check is a guard-rail, not an evaluation.** Real reasoning-quality
  evaluation needs human artefact annotations (Skyra used ViF-CoT-4K) which this
  project does not have.

## References

- Skyra — *AI-Generated Video Detection via Grounded Artifact Reasoning*, arXiv 2512.15693
- Effort — *Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection*, ICML 2025
- DF40 — *Toward Next-Generation Deepfake Detection*, NeurIPS 2024 D&B
- DeepfakeBench, NeurIPS 2023 — arXiv 2307.01426
- AltFreezing for More General Video Face Forgery Detection, CVPR 2023
- LFWS / LFWL — lightweight wavelet + phase/LBP fusion, May 2026
- Fridrich & Kodovsky, *Rich Models for Steganalysis of Digital Images*, 2012
- Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition*, 2020


---

## The frontend

```bash
streamlit run app/streamlit_app.py
```

Upload a video, get a verdict — and six tabs of *why*:

| tab | question | method |
|---|---|---|
| Summary | what happened | plain-English bullets, each traceable to a number |
| Where | which pixels | Grad-CAM over the spatial branch |
| When | which frames | per-frame occlusion, `freeze` or `blank` mode |
| Which branch | what the answer depended on | branch ablation vs the attention gate |
| Forensic measurements | which quantity was abnormal | percentile-calibrated TSC descriptors + the TSC map |
| Reasoning | the written analysis | Qwen2.5-VL if installed, else a template that cannot hallucinate |

Two deliberate choices: the verdict is **not** the biggest thing on the page (a
detector at 0.83 AUC is wrong about one video in six, and a huge red FAKE trains
users to stop reading), and crop size is **not** a user setting — the forensic
branches use a fixed radial-ring index, so the pipeline adopts the checkpoint's
resolution and says so.

---

## Explainability — `src/fsvnet/explain.py`

Four methods, each answering a different question, and each honest about its
limits:

* **`grad_cam`** — *where* in the frame. Spatial branch only; the forensic
  branches have no meaningful spatial map and inventing one would be decoration.
* **`temporal_occlusion`** — *when* in the clip. Default `freeze` mode replaces
  frame *t* with frame *t−1* rather than blanking it: blanking creates an
  artificial discontinuity that makes every frame look important to a temporal
  model.
* **`branch_attribution`** — *which branch the answer depended on*. This is the
  honest version of the attention gate. The gate says what the fusion scaled up;
  it does not say whether the logit would move if that branch were removed. This
  zeroes each branch and measures the shift, and **flags where the two
  disagree** — that disagreement is worth a sentence in the write-up.
* **percentile-calibrated descriptors** — *which measured quantity was abnormal*,
  relative to the training reals.

---

## The Qwen-VL detector path — `src/fsvnet/qwen_detector.py`

Qwen2.5-VL fine-tuned with QLoRA as the detector **itself**, with the TSC
measurements injected into the prompt:

```
<image> <image> <image> <image>
Forensic measurements for this clip (percentile = position among real videos):
  temporal_whiteness   +0.4120   (3rd pct of real)   <-- ANOMALOUS
  temporal_peakiness   +0.3100   (97th pct of real)  <-- ANOMALOUS
Is this video real or manipulated? Answer with one word.
Answer:
```

The verdict is read by **constrained-token scoring** — the logits of ` real` and
` fake` at the answer position, softmaxed over just those two. That gives a
calibrated probability for the ROC curve, is differentiable so it doubles as the
training objective, and keeps the model generative for the written rationale.

**Why this is not just Skyra again:** a VLM cannot see a temporal spectral
anomaly. It lives in the relationship between frames at a frequency scale no
tokeniser represents. Handing the model the measurement is the only way it can
use that evidence — which makes the ablation clean:

| variant | what it is |
|---|---|
| `qwen_frames_only` | the Skyra-style baseline |
| `qwen_frames_plus_tsc` | this work |
| `qwen_tsc_only` | measurements with no pixels — the control |

The first two differ in exactly one flag, so the gap between them *is* the value
of the forensic channel, measured rather than asserted.

```python
from fsvnet.qwen_detector import QwenDetectorConfig, preflight
preflight(QwenDetectorConfig())      # run this BEFORE a training job, not after
```

**Status:** everything that does not need model weights is tested
(`python scripts/test_qwen_path.py`, 23/23). Weight loading, LoRA attachment and
training are **not** tested here — this environment has no Hugging Face access,
so the first real run happens on your machine. `preflight()` is the substitute.

8 GB is tight but workable: 3B in NF4 ≈ 2.2 GB, vision tower **frozen** (LoRA on
the language model only — unfreezing it is what turns tight into OOM), 4 frames,
gradient checkpointing, batch 1 × accumulation 16 → ~6–7 GB, ~2–4 s/step.

---

## Comparison against Skyra — `src/fsvnet/benchmark.py`

```python
from fsvnet.benchmark import print_capability_report, run_comparison, check_protocol
print_capability_report()
```

**What this module will not do:** produce a table claiming we beat Skyra.
Skyra's weights are not released, we have no numbers on any shared benchmark,
and the two systems do not perform the same task. A number invented for that
cell is the fastest way to lose an examiner's trust.

**What it does instead:**

1. `capability_table()` — a qualitative, cited positioning against Skyra and
   Effort. Every cell is a published fact or a property of our own code; the
   "reported benchmark" row says *NOT YET MEASURED* for us, deliberately.
2. `run_comparison()` — a quantitative table against baselines you can actually
   execute (`frame_cnn`, `fsw_net_video`, `tsc_only`, `fsv_net_full`), all on one
   identity-disjoint split, with a **paired bootstrap** between models.

The paired test matters: two models evaluated on the same videos are not
independent samples, so comparing their separate CIs is the wrong test.
`paired_bootstrap` resamples the videos jointly and reports the distribution of
the *difference*, which is what lets you write "0.83 vs 0.79, p = 0.04" instead
of hoping nobody asks. When it returns `not resolved`, say so — that is a
finding, not a gap to fill with a bigger claim.
