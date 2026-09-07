# FSW-Net — Frequency-Spatial-Watermark Network for Deepfake & AI-Image Detection

A capstone project that detects whether an image (or video frame) is **real**,
**deepfake**, **edited**, or **ai_generated** using a novel tri-branch
architecture and explains its decisions in natural language.

> **What's novel:** FSW-Net fuses three complementary forensic signals —
> spatial semantics (EfficientNet-B0), 2-D FFT frequency artifacts, and SRM
> watermark/residual traces — through a learnable branch-attention gate. A
> Claude-vision module then narrates the evidence the model used.

---

## 1. Repository Layout

```
DeepfakeProject/
├── data/
│   ├── raw/                # <-- DROP YOUR SOURCE DATA HERE
│   │   ├── real/
│   │   ├── deepfake/
│   │   ├── edited/
│   │   └── ai_generated/
│   ├── processed/          # auto-built train/val/test splits
│   │   ├── train/{real,deepfake,edited,ai_generated}/
│   │   ├── val/  {…}
│   │   └── test/ {…}
│   ├── balanced/           # class-balanced + augmented training set
│   │   └── train/{…}
│   └── metadata/           # optional CSV manifests
├── scripts/
│   └── prepare_dataset.py  # raw → processed splits + video-frame extraction
├── src/
│   ├── fsw_net.py          # FSW-Net model definition + FocalLoss
│   ├── explainability.py   # Claude-vision forensic explainer + figure builder
│   └── inference_pipeline.py # UI-agnostic single/batch/video inference engine
├── notebooks/
│   ├── 03_build_balanced_dataset.ipynb          # step 2
│   ├── balanced_training_clean.ipynb            # step 3a — EfficientNet baseline
│   ├── proposed_model_training.ipynb            # step 3b — alt baseline
│   ├── fsw_novel_detector.ipynb                 # step 4 — train FSW-Net
│   ├── model_evaluation_and_explainability.ipynb# step 5
│   ├── model_comparison.ipynb                   # step 5
│   ├── fsw_explainable_inference.ipynb          # step 6 — run inference + Claude
│   ├── trial_and_learn.ipynb                    # scratch
│   ├── best_fsw_net.pth                         # ★ final trained model
│   ├── best_efficientnet_b0.pth                 # baseline ckpt
│   ├── best_balanced_efficientnet_b0.pth        # balanced baseline ckpt
│   ├── best_acc_model.pth                       # ckpt selected by val accuracy
│   └── best_loss_model.pth                      # ckpt selected by val loss
├── checkpoints/            # additional model checkpoints (if any)
└── outputs/
    ├── predictions/        # CSVs / JSON written by inference runs
    └── plots/              # confusion matrix, FFT spectra, SRM residuals…
```

---

## 2. Requirements

- **Python:** 3.10 or 3.11
- **GPU:** strongly recommended (CUDA 11.8+ / RTX-class card). CPU works for
  inference but training will be slow.
- **Disk:** plan for 5–20 GB depending on dataset size.
- **OS:** Windows, Linux, or macOS.

### Core Python packages

```
torch>=2.0
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
Pillow
tqdm
ipywidgets
jupyter
anthropic            # for natural-language explanations
```

Install everything in one go (see the next section).

---

## 3. Quick Start

### 3.1 Clone / unzip the project, then create a virtual environment

```bash
cd DeepfakeProject

# create env
python -m venv .venv
# activate it
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   macOS / Linux:
source .venv/bin/activate

# upgrade pip and install deps
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python \
            Pillow tqdm ipywidgets jupyter anthropic
```

> If you don't have a CUDA GPU, drop the `--index-url` flag — pip will install
> the CPU build of PyTorch.

### 3.2 Set your Anthropic API key (only needed for natural-language explanations)

```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also paste the key directly into the inference notebook's config cell.

### 3.3 Just want to **run inference** with the trained model?

Skip ahead to **Step 6** below. A pre-trained checkpoint
(`notebooks/best_fsw_net.pth`) is included.

---

## 4. The Pipeline — Run These In Order

Each step is either a Python script or a Jupyter notebook. Launch Jupyter once:

```bash
jupyter notebook
```

…and open the notebooks from the `notebooks/` folder.

### Step 1 — Prepare the dataset

**Script:** `scripts/prepare_dataset.py`

1. Drop your source media into `data/raw/<class>/`, one folder per class:
   `real/`, `deepfake/`, `edited/`, `ai_generated/`.
   Supported formats: `.jpg .jpeg .png .bmp .webp` for images and
   `.mp4 .avi .mov .mkv .webm` for videos. Videos are sampled into 8 frames
   each by default (configurable at the top of the script).
2. From the project root, run:
   ```bash
   python scripts/prepare_dataset.py
   ```
3. Output: `data/processed/{train,val,test}/<class>/…` populated with images.
   Default split is 70 / 15 / 15 with `RANDOM_SEED = 42` for reproducibility.

### Step 2 — Build a class-balanced training set

**Notebook:** `notebooks/03_build_balanced_dataset.ipynb`

Some classes will have far more samples than others. This notebook
oversamples small classes with mild augmentation (flip / rotate / brightness /
contrast / sharpness / colour) so every class hits the same target count
(default 400). Output goes to `data/balanced/train/<class>/`.

Run all cells. Confirm class counts are equal at the end.

### Step 3 — (Optional) Train the EfficientNet baselines

These give you a single-branch reference point to compare FSW-Net against.

| Notebook | What it produces |
|---|---|
| `notebooks/balanced_training_clean.ipynb` | `best_balanced_efficientnet_b0.pth` |
| `notebooks/proposed_model_training.ipynb` | `best_efficientnet_b0.pth`, `best_acc_model.pth`, `best_loss_model.pth` |

Run all cells. Training time on a single mid-range GPU is roughly 10–30 min
each at the default epoch count.

### Step 4 — Train FSW-Net (the proposed novel model)

**Notebook:** `notebooks/fsw_novel_detector.ipynb`

This trains the tri-branch network defined in `src/fsw_net.py`:

- **Branch 1 — Spatial:** EfficientNet-B0 (ImageNet-pretrained)
- **Branch 2 — Frequency:** 2-D FFT log-magnitude CNN + 64-bin radial power
  spectrum
- **Branch 3 — Watermark:** six fixed SRM high-pass kernels → CNN
- **Fusion:** learnable softmax attention over the three branch embeddings
- **Loss:** Focal loss with optional per-class weights

The notebook saves the best checkpoint to `notebooks/best_fsw_net.pth` and
writes training curves to `outputs/plots/fsw_training_curves.png`.

### Step 5 — Evaluate & compare models

| Notebook | Purpose |
|---|---|
| `notebooks/model_evaluation_and_explainability.ipynb` | Test-set metrics, confusion matrices, per-class precision/recall, sample explanations |
| `notebooks/model_comparison.ipynb` | Side-by-side FSW-Net vs. EfficientNet baselines |

These read the checkpoints saved in Steps 3–4. Run all cells. Figures are
written to `outputs/plots/`.

### Step 6 — Run explainable inference on your own media

**Notebook:** `notebooks/fsw_explainable_inference.ipynb`

The recommended entry point for end users. In one notebook you can:

1. Classify a **single image**
2. Classify a **folder of images** (batch) with a summary table
3. Classify a **video** — frames are sampled, voted, and the most confident
   frame is explained

Each mode optionally produces a Claude-generated forensic narrative backed by
three figures: SRM residual maps, FFT spectrum, and a decision-analysis bar
chart (probabilities + branch attention weights).

Configure two values in **Cell 2** of the notebook:

```python
MODEL_PATH = "best_fsw_net.pth"                 # relative to notebooks/
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
```

Then run cells top-to-bottom and use the interactive widgets to pick a file.

---

## 5. Programmatic Inference (no notebook)

`src/inference_pipeline.py` exposes a clean API that any front-end (Streamlit,
Gradio, FastAPI, CLI, etc.) can wrap.

```python
import sys; sys.path.insert(0, "src")
from inference_pipeline import FSWInferencePipeline

pipeline = FSWInferencePipeline(
    model_path        = "notebooks/best_fsw_net.pth",
    class_names       = ["ai_generated", "deepfake", "edited", "real"],
    anthropic_api_key = "sk-ant-...",      # or leave None to read env var
)

# Single image
result = pipeline.run_single("path/to/image.jpg", explain=True)
print(result["predicted_class"], result["confidence"])
print(result.get("explanation"))

# Batch
results = pipeline.run_batch("path/to/folder/", explain=True, max_explain=5)

# Video
video = pipeline.run_video("clip.mp4", frame_step=24, explain=True)
print(video["majority_class"], video["confidence"])
```

`class_names` **must** match the order used during training. Because the
training notebooks use `torchvision.datasets.ImageFolder`, that ordering is
alphabetical: `["ai_generated", "deepfake", "edited", "real"]`.

---

## 6. Extending the Project

A few good directions:

- **Add a new class** (e.g., `face_reenactment`): create
  `data/raw/face_reenactment/`, rerun Steps 1–4, and bump `num_classes` when
  instantiating `FSWNet(num_classes=5, …)` in
  `notebooks/fsw_novel_detector.ipynb`.
- **Plug in a different spatial backbone:** swap the `efficientnet_b0` call
  inside `FSWNet.__init__` (in `src/fsw_net.py`) for any torchvision model
  that exposes a `.features` attribute (ConvNeXt, EfficientNet-V2, etc.).
  Update the `1280` projection dim to match the new backbone's output
  channels.
- **Add a fourth forensic branch** (e.g., color-correlation or DCT residuals):
  follow the pattern of `FrequencyBranch` / `SRMWatermarkBranch` and pass it
  through `BranchAttentionFusion` with `num_branches=4`.
- **Replace the explainer LLM:** `src/explainability.py` is the only file
  that imports `anthropic`. Swap the API call for any vision-capable model
  while keeping the figure builder intact.
- **Ship a web UI:** import `FSWInferencePipeline` from a Streamlit or Gradio
  app — every method already returns plain dicts and PIL images.
- **Try longer / wider training:** edit the hyper-parameter cell at the top
  of `fsw_novel_detector.ipynb` (`BATCH`, `EPOCHS`, `LR`, focal-loss
  `gamma`).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Reduce `BATCH` in the training notebook (try 16 → 8). |
| `FileNotFoundError: .../best_fsw_net.pth` | Run Step 4, or update `MODEL_PATH` in the inference notebook. |
| Predictions look random | Confirm `class_names` order matches training (alphabetical from `ImageFolder`). |
| `anthropic.AuthenticationError` | `ANTHROPIC_API_KEY` not set — see Section 3.2. |
| `cv2` import fails | `pip install opencv-python` (or `opencv-python-headless` on servers). |
| Notebook can't find `src` modules | The inference notebook adds `../src` to `sys.path` automatically; if you run scripts from elsewhere, do the same. |
| Class imbalance hurting metrics | Make sure Step 2 (`03_build_balanced_dataset.ipynb`) actually ran and `data/balanced/train` is populated. |

---

## 8. Reproducibility Notes

- All training notebooks seed Python, NumPy, and PyTorch with `SEED = 42`.
- `prepare_dataset.py` uses `RANDOM_SEED = 42` for the train/val/test split.
- Checkpoints are saved in `notebooks/` (and any `checkpoints/` you create).
  Keep them with the code if you want exact reproductions.

---

## 9. Credits & References

- Fridrich & Kodovsky, *Rich Models for Steganalysis of Digital Images* (2012)
- Corvi et al., *On the Detection of Synthetic Images Generated by Diffusion Models* (2022)
- Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition* (2020)
- Tan & Le, *EfficientNet* (2019)
- Anthropic Claude — vision-language explanations
