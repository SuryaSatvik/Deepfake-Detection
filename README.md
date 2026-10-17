# FSV-Net/DFE — dual forensic evidence for face-swap video

Face-swap detection on Celeb-DF v2, with two forensic measurements that no
standard video detector computes, and an evaluation protocol built to stop the
one mistake that inflates most published numbers on this dataset.

**Surya Satvik** · B.Tech, Artificial Intelligence and Machine Learning,
PES University, Bangalore · capstone project, 2026

> Read [`docs/FINDINGS.md`](docs/FINDINGS.md) before you read any number
> anywhere else in this repository. It contains the measured results —
> including a negative one — that constrain every claim here.

---

## Run it in five minutes, with no dataset

You do not need Celeb-DF v2 to try the app. Clone, install, and drop in any
video with a face in it.

```bash
git clone https://github.com/SuryaSatvik/Deepfake-Detection.git
cd Deepfake-Detection

# 1. torch first, matched to your hardware
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   ... or, with an NVIDIA card:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. everything else
pip install -r requirements.txt

# 3. the trained weights (~145 MB, from the release — not in git)
python scripts/download_weights.py

# 4. go
streamlit run streamlit/app.py
```

Python 3.10 or 3.11. A GPU is optional for the app and necessary only for
training.

### What you get, and what you don't

An uploaded video carries **no 81-point facial landmarks**, and the LMF branch
is built on them. So on an upload the app runs the two conditions that don't
need landmarks (`B0` conventional and `C1` + TSC) and **explicitly disables**
`C2` and `C3` rather than feeding the branch zeros and reporting a verdict
anyway. Full five-branch inference needs a Celeb-DF v2 clip, or your own
landmark extraction.

Faces are detected with `mediapipe` if it is installed, otherwise with the Haar
cascade that ships inside `opencv-python`, otherwise a centre crop. The app
reports which backend ran and in what fraction of frames a face was actually
found, because a low ratio means the crops are wrong and the verdict is
meaningless. `pip install mediapipe` for a much better detector.

With no checkpoint present at all the app still runs, in **evidence-only mode**:
it computes and displays the forensic measurements and states plainly that
there is no detector behind them.

---

## The 60-second version of the research

| | |
|---|---|
| **Dataset** | Celeb-DF v2 as pre-extracted 256×256 face crops + 81-point landmarks. 6,526 clips, 304 identity groups, 86.4% fake. |
| **Split** | Identity-disjoint via union-find over every identity pair a swap touches. A random split would leak **95.1%** of the test set; the official Celeb-DF v2 test list leaks **96.5%**. |
| **Contribution** | **TSC** (temporal spectral coherence) + **LMF** (landmark motion-compensated forensics), as percentile-calibrated measurements rather than opaque features. |
| **Honesty** | The pre-registered CPU probe of both channels came back near chance. That is reported as the result, not buried. |

### TSC — temporal spectral coherence

In real footage the finest spatial detail is dominated by photon and read
noise, near-independent frame to frame, so the fine radial rings of the power
spectrum fluctuate like *white noise along the time axis*. In synthesised or
swapped video that detail is computed from a temporally smooth latent, so it
becomes temporally *coloured* — over-smooth — or *spiked* where blending and
colour correction intervene.

Per frame: azimuthal power spectrum `A ∈ R^{T×K}`; log-compress; subtract the
temporal mean and linear drift per ring; Hann-window; rFFT along time →
`C ∈ R^{F×K}`. A per-frame detector cannot see this: the quantity lives in the
relationship *between* frames.

### LMF — landmark motion-compensated forensics

In image coordinates a pixel changes mostly because the head moved, so any
temporal forensic statistic is dominated by motion. LMF estimates a per-frame
Procrustes similarity from the 81 landmarks (using only motion-stable points —
eye corners and the nose bridge), warps into a canonical face frame, and takes
the residual `R_t = Ĩ_t − Ĩ_{t−1}` **there**. Head motion is removed
*geometrically* rather than learned away.

Because the warp fixes the coordinate frame, the face mask covers the same
pixels in every frame, so residual energy pools over three regions that mean
the same thing throughout a clip: **inner face** (what a swap rewrites),
**blend-seam band**, and **outside** (what it leaves alone). The contrast
between them measures the asymmetry that defines a face swap.

The two channels share no operator and no coordinate frame, and both read from
a cache built before training — so `−TSC` / `−LMF` change the architecture
without changing the compute budget, which is what makes the ablation a
comparison of designs rather than of training budgets.

---

## Trained weights

Not in git. Four checkpoints at ~37 MB each would bloat every clone for the
many people who only want to read the code, so they are attached to a
[release](https://github.com/SuryaSatvik/Deepfake-Detection/releases) and fetched by
`scripts/download_weights.py`.

| Condition | Branches | Landmarks needed at inference |
|---|---|---|
| `B0` | spatial + frequency + SRM | no |
| `C1` | `B0` + TSC | no |
| `C2` | `B0` + LMF | **yes** |
| `C3` | `B0` + TSC + LMF — **the proposed model** | **yes** |

All four are published so the ablation can be reproduced, not only the demo.

---

## Reproducing the research

The dataset is gated: Celeb-DF v2 requires a signed request to its authors at
[yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics).
It is not redistributed here and nothing in this repository can automate that
step.

With the data in `Celeb-DF-v2/`:

```bash
python scripts/build_celebdf_manifest.py            # manifest + identity split, ~2 min
python scripts/extract_lmf.py --frames 16 --save-energy   # forensic cache, ~45 min, resumable
python scripts/train_conditions.py                  # the four conditions, GPU
python scripts/eval_conditions.py                   # test-split metrics + ablation
python scripts/make_explainability.py               # Grad-CAM, occlusion, attribution
jupyter notebook "notebooks/model analysis.ipynb"   # the tables and figures
```

Training belongs in a terminal, not in Jupyter — three runs were lost to a
stale in-browser copy, to host RAM exhausted by notebook plus workers, and to
an interrupted cell losing its return value. The notebook's job is loading
finished checkpoints and producing tables.

`python scripts/verify_project.py` asserts the properties this repository
claims, including that no reported number was hand-typed.

---

## Layout

```
├── src/fsvnet/
│   ├── lmf.py  tsc.py  tsc_np.py    the two forensic channels
│   ├── celebdf_frames.py            adapter for the pre-extracted distribution
│   ├── dataset_frames.py            loader for frames + landmarks + LMF cache
│   ├── model.py                     FSV-Net/DFE, 5 switchable branches
│   ├── manifest.py                  identity-disjoint splitting (union-find)
│   ├── faces.py                     face detection for arbitrary uploads
│   ├── evidence.py  explain.py      evidence pack, Grad-CAM, occlusion
│   ├── qwen_detector.py             Qwen2.5-VL as detector, QLoRA
│   ├── train.py  evaluate.py        training, video-level metrics, ablations
│   └── cfx.py + calibrate, uncertainty, selective, attribution,
│       faithfulness, arbitrate, narrate, xai_eval
│                                    CFX explanation layer (docs/CFX.md)
├── streamlit/app.py                 the frontend
├── streamlit/cfx_app.py             the CFX frontend (verdict, confidence, one explanation)
├── assets/lexicon.yaml              technical/layman vocabulary for narration
├── notebooks/model analysis.ipynb   the analysis, 24 sections
├── notebooks/CFX_RUN_ALL.ipynb      fit + evaluate the CFX layer end to end
├── notebooks/XAI_Eval_RESULTS.ipynb XAI evaluation results
├── scripts/                         manifest, extractors, training, evaluation
├── results/                         metrics, predictions, ablations, figures
├── image_stage/                     the earlier image-level project, preserved
├── docs/FINDINGS.md                 the measured results — read this first
└── docs/CFX.md                      the explanation layer, implementation reference
```

---

## What this does not claim

* **Within-dataset only.** Cross-dataset transfer to FaceForensics++ or DFDC
  has not been run. That number, not the in-domain one, is what would say
  anything about generalisation.
* **Compression untested.** Media distributed online is re-encoded, and
  detectors of this family are known to collapse under aggressive JPEG.
* **No audio.** The Celeb-DF v2 derivative ships none, no audio model was
  trained, and the app refuses audio uploads with a reason rather than
  offering a capability it does not have.
* **The forensic channels are a hypothesis with a mechanism and a
  pre-registered test.** On closed-form descriptors alone both sit near chance
  across unseen identities (`docs/FINDINGS.md` §3). The learned branches see
  far more than those descriptors, but the contribution is established by the
  ablation or not at all.
* **Skyra is not a baseline here.** It is trained on fully generated video and
  its artefact taxonomy has no category for localised face replacement, so
  running it on Celeb-DF is an out-of-distribution transfer test, not a
  like-for-like comparison.

---

## Credit and citation

Built by **Surya Satvik** (PES University, Bangalore) as a B.Tech capstone in
Artificial Intelligence and Machine Learning, 2026.

```bibtex
@software{satvik2026fsvnetdfe,
  author = {Satvik, Surya},
  title  = {FSV-Net/DFE: Dual Forensic Evidence for Face-Swap Video Detection},
  year   = {2026},
  url    = {https://github.com/SuryaSatvik/Deepfake-Detection}
}
```

Code is MIT licensed — see [`LICENSE`](LICENSE). The Celeb-DF v2 dataset is
not, and is not redistributed here.

## References

- Celeb-DF v2 — Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020
- Skyra — *AI-Generated Video Detection via Grounded Artifact Reasoning*, [arXiv 2512.15693](https://arxiv.org/abs/2512.15693), CVPR 2026
- Qwen2.5-VL — [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- DeepfakeBench, NeurIPS 2023 — [arXiv 2307.01426](https://arxiv.org/abs/2307.01426)
- AltFreezing for More General Video Face Forgery Detection, CVPR 2023
- Fridrich & Kodovský, *Rich Models for Steganalysis of Digital Images*, 2012
