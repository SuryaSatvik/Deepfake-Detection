"""
FSW-Net Explainability Module
==============================
Uses Claude's vision API to generate natural-language forensic explanations
for each prediction, backed by three visual evidence images:

  1. SRM Residual Maps  — the noise patterns the model's watermark branch read
  2. FFT Spectrum       — frequency artifacts from the model's frequency branch
  3. Decision Analysis  — class probabilities + branch attention weights

The module is UI-framework-agnostic: it returns PIL Images and plain strings
so callers (notebook, Streamlit, Gradio, etc.) can render them however they like.
"""

import io
import base64
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

_SRM_FILTER_NAMES = [
    "Horiz\nDiff", "Vert\nDiff", "Diag\nDiff",
    "Anti-Diag\nDiff", "Local Avg\nResidual", "Laplacian",
]

_BRANCH_LABELS = ["Spatial\n(EfficientNet)", "Frequency\n(FFT)", "Watermark\n(SRM)"]
_BRANCH_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

_CLASS_RISK = {
    "real":         "LOW  — image appears authentic",
    "edited":       "MEDIUM — minor manipulation detected",
    "deepfake":     "HIGH — face/body swap detected",
    "ai_generated": "HIGH — fully synthetic image detected",
}


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """ImageNet-denormalise a (1,3,H,W) tensor → (H,W,3) float32 in [0,1]."""
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Evidence figure builders
# ──────────────────────────────────────────────────────────────────────────────

def build_srm_figure(srm_maps: np.ndarray) -> Image.Image:
    """
    Render the 6 primary SRM residual maps (one per forensic kernel).

    srm_maps : (18, H, W) numpy — raw output of SRMWatermarkBranch.srm_conv
               (6 kernels × 3 input channels; we display channel-0 of each kernel)
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()

    for k in range(6):
        r = srm_maps[k]                         # channel-0 of k-th kernel
        vmax = max(np.percentile(np.abs(r), 98), 1e-6)
        axes[k].imshow(r, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[k].set_title(_SRM_FILTER_NAMES[k], fontsize=9, pad=4)
        axes[k].axis("off")

    fig.suptitle(
        "SRM Residual Maps — Forensic Watermark Analysis\n"
        "Red = positive residual  |  Blue = negative residual  |  "
        "Structured patterns → manipulation",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    return _fig_to_pil(fig)


def build_spectrum_figure(freq_branch, img_tensor: torch.Tensor) -> Image.Image:
    """
    Render the FFT log-magnitude heatmap and azimuthal power spectrum.
    """
    with torch.no_grad():
        log_mag   = freq_branch._fft_log_magnitude(img_tensor)   # (1,3,H,W)
        azimuthal = freq_branch._azimuthal_mean(img_tensor)       # (1, BINS)

    log_mag_np = log_mag[0, 0].cpu().numpy()           # single channel
    spec_np    = azimuthal[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # FFT heatmap
    im = axes[0].imshow(log_mag_np, cmap="inferno")
    axes[0].set_title("FFT Log-Magnitude Spectrum\n(bright = high energy)", fontsize=10)
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Azimuthal power spectrum
    x = np.linspace(0, 0.5, len(spec_np))
    axes[1].plot(x, spec_np, color="royalblue", linewidth=2)
    axes[1].fill_between(x, 0, spec_np, alpha=0.25, color="royalblue")
    axes[1].set_xlabel("Normalised Spatial Frequency (cycles/pixel)", fontsize=9)
    axes[1].set_ylabel("Log Power", fontsize=9)
    axes[1].set_title(
        "Azimuthal Power Spectrum\n"
        "Real cameras: smooth 1/f² roll-off  |  AI images: bumps / elevated floor",
        fontsize=9,
    )
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Frequency Domain Analysis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _fig_to_pil(fig)


def build_decision_figure(
    probs: np.ndarray,
    branch_weights: np.ndarray,
    class_names: list,
    prediction: int,
) -> Image.Image:
    """
    Render class probabilities and branch attention weights side-by-side.
    """
    pred_name  = class_names[prediction]
    bar_colors = [
        "#e74c3c" if c == pred_name else "#7fb3d3" for c in class_names
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # — Class probabilities —
    bars = axes[0].barh(class_names, probs * 100, color=bar_colors, alpha=0.88)
    axes[0].set_xlabel("Confidence (%)", fontsize=9)
    axes[0].set_title("Classification Probabilities", fontsize=10)
    axes[0].set_xlim(0, 110)
    for bar, val in zip(bars, probs):
        axes[0].text(
            val * 100 + 1, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", fontsize=9,
        )

    # — Branch attention —
    h_bars = axes[1].barh(_BRANCH_LABELS, branch_weights,
                           color=_BRANCH_COLORS, alpha=0.88)
    axes[1].set_xlabel("Attention Weight", fontsize=9)
    axes[1].set_title("Branch Trust  (what the model relied on)", fontsize=10)
    axes[1].set_xlim(0, 0.85)
    for bar, val in zip(h_bars, branch_weights):
        axes[1].text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    risk = _CLASS_RISK.get(pred_name, "UNKNOWN")
    fig.suptitle(
        f"Decision: {pred_name.upper()}  ({probs[prediction]*100:.1f}% confidence)"
        f"\nRisk Level: {risk}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    return _fig_to_pil(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main explainer
# ──────────────────────────────────────────────────────────────────────────────

class FSWExplainer:
    """
    Generates an LLM-powered forensic explanation for an FSW-Net prediction.

    Parameters
    ----------
    api_key  : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    model    : Claude model to use for explanations
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)   # uses env var if None
        self.model  = model or self.DEFAULT_MODEL

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        class_names: list,
        prediction: int,
        probs: np.ndarray,
        branch_weights: np.ndarray,
    ) -> str:
        pred_name = class_names[prediction]
        prob_lines = "\n".join(
            f"  • {class_names[i]}: {probs[i]*100:.1f}%"
            for i in range(len(class_names))
        )
        bw_lines = "\n".join(
            f"  • {lbl.replace(chr(10),' ')}: {w:.3f}"
            for lbl, w in zip(_BRANCH_LABELS, branch_weights)
        )
        dominant = _BRANCH_LABELS[int(branch_weights.argmax())].replace("\n", " ")
        risk      = _CLASS_RISK.get(pred_name, "UNKNOWN")

        return textwrap.dedent(f"""
            You are a senior forensic AI analyst reviewing the output of FSW-Net,
            a tri-branch neural network that detects AI-generated and manipulated images.

            ── PREDICTION ──────────────────────────────────────────────
            Class predicted : {pred_name.upper()}
            Confidence      : {probs[prediction]*100:.1f}%
            Risk level      : {risk}

            ── CLASS PROBABILITIES ─────────────────────────────────────
            {prob_lines}

            ── BRANCH ATTENTION WEIGHTS ────────────────────────────────
            (Higher weight = the model relied more on this forensic branch)
            {bw_lines}
            Dominant branch : {dominant}

            ── BRANCH DESCRIPTIONS ─────────────────────────────────────
            • Spatial (EfficientNet-B0) — analyses visible textures, colours, and
              semantic content learned from millions of real photographs.
            • Frequency (FFT)           — analyses the 2-D power spectrum; GAN
              upsampling creates periodic checkerboard peaks; diffusion models
              elevate the high-frequency noise floor.
            • Watermark (SRM)           — applies 6 fixed forensic high-pass kernels
              (horizontal/vertical/diagonal difference, Laplacian, local-average
              residual) to reveal microscopic noise residuals that generative models
              leave behind — analogous to invisible watermarks.

            ── ATTACHED IMAGES (in order) ──────────────────────────────
            Image 1 — Original input image.
            Image 2 — SRM residual maps (6 panels): red = positive residual,
                       blue = negative. Structured / periodic patterns indicate
                       manipulation; random noise indicates authenticity.
            Image 3 — FFT analysis: left = log-magnitude heatmap (bright = high
                       energy), right = azimuthal power spectrum. Bumps or an
                       elevated high-frequency floor suggest AI generation.
            Image 4 — Decision chart: class probabilities (left) and branch
                       attention weights (right).

            ── YOUR TASK ────────────────────────────────────────────────
            Write a clear forensic explanation (4-6 paragraphs) covering:

            1. State the verdict and confidence in plain language.
            2. Identify which forensic branch drove the decision and what that
               implies about where the manipulation/generation fingerprint lies.
            3. Describe specific visual observations in the SRM residual maps
               (Image 2) that support the prediction — are the residuals structured
               or random? where is energy concentrated?
            4. Describe what the frequency spectrum (Image 3) reveals — does the
               power spectrum deviate from the expected 1/f² roll-off? are there
               unusual peaks?
            5. Note any uncertainty, borderline probabilities, or caveats the user
               should be aware of.

            Write for a technical-but-non-specialist audience (e.g. a journalism
            ethics team or a digital forensics student). Be specific and reference
            the visual evidence.
        """).strip()

    # ------------------------------------------------------------------
    def explain(
        self,
        model,                           # FSWNet instance
        img_tensor: torch.Tensor,        # (1, 3, 224, 224) on device
        prediction: int,
        probs: np.ndarray,
        branch_weights: np.ndarray,
        class_names: list,
    ) -> dict:
        """
        Run full LLM explanation for one prediction.

        Returns
        -------
        dict with keys:
          explanation  : str — LLM-generated forensic narrative
          figures      : dict[str, PIL.Image] — evidence images
          prediction   : str
          confidence   : float
          probs        : dict[str, float]
          branch_weights : dict[str, float]
          risk_level   : str
        """
        # ── Build evidence figures ─────────────────────────────────────
        orig_np   = _denorm(img_tensor)
        orig_pil  = Image.fromarray((orig_np * 255).astype(np.uint8))

        srm_maps  = model.get_srm_residuals(img_tensor)[0].cpu().numpy()
        fig_srm   = build_srm_figure(srm_maps)
        fig_spec  = build_spectrum_figure(model.freq_branch, img_tensor)
        fig_dec   = build_decision_figure(probs, branch_weights, class_names, prediction)

        # ── Call Claude API ────────────────────────────────────────────
        prompt = self._build_prompt(class_names, prediction, probs, branch_weights)

        def _img_block(pil_img: Image.Image) -> dict:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _pil_to_b64(pil_img),
                },
            }

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    _img_block(orig_pil),
                    _img_block(fig_srm),
                    _img_block(fig_spec),
                    _img_block(fig_dec),
                ],
            }],
        )
        explanation = response.content[0].text

        # ── Return structured result ───────────────────────────────────
        bw_dict = {
            lbl.replace("\n", " "): float(w)
            for lbl, w in zip(_BRANCH_LABELS, branch_weights)
        }
        return {
            "prediction":     class_names[prediction],
            "confidence":     float(probs[prediction]),
            "probs":          {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            "branch_weights": bw_dict,
            "risk_level":     _CLASS_RISK.get(class_names[prediction], "UNKNOWN"),
            "explanation":    explanation,
            "figures": {
                "original":          orig_pil,
                "srm_residuals":     fig_srm,
                "frequency_spectrum": fig_spec,
                "decision_analysis": fig_dec,
            },
        }
