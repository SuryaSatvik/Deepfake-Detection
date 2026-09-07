"""
FSW-Net Inference Pipeline
===========================
UI-framework-agnostic inference engine.  Supports three input modes:

  single  — one image file (jpg, png, webp, …)
  batch   — a folder of images → returns a list of results + summary
  video   — a video file → frame sampling → majority-vote + best-frame explanation

All public methods return plain dicts / lists of dicts so they can be consumed
by any front-end (Streamlit, Gradio, Jupyter, REST API, etc.).

Usage
-----
    pipeline = FSWInferencePipeline(
        model_path   = "notebooks/best_fsw_net.pth",
        class_names  = ["ai_generated", "deepfake", "edited", "real"],
        anthropic_api_key = "sk-ant-...",
    )

    # single image
    result = pipeline.run_single("path/to/image.jpg", explain=True)

    # batch folder
    results = pipeline.run_batch("path/to/folder/", explain=True, max_explain=5)

    # video
    result = pipeline.run_video("path/to/video.mp4", frame_step=24, explain=True)
"""

import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fsw_net import FSWNet
from explainability import FSWExplainer


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SUPPORTED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class FSWInferencePipeline:
    """
    Parameters
    ----------
    model_path        : path to .pth checkpoint saved by FSWNet
    class_names       : list in the same order as training (ImageFolder ordering)
    device            : torch device; auto-detected if None
    anthropic_api_key : Anthropic key for LLM explanations; uses env var if None
    llm_model         : Claude model ID for explanations
    """

    def __init__(
        self,
        model_path: str,
        class_names: list,
        device=None,
        anthropic_api_key: str | None = None,
        llm_model: str | None = None,
    ):
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        # Load model
        self.model = FSWNet(num_classes=len(class_names)).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded from {model_path}  |  device: {self.device}")

        # Explainer (lazy – only instantiated when explain=True)
        self._api_key   = anthropic_api_key
        self._llm_model = llm_model
        self._explainer: Optional[FSWExplainer] = None

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_explainer(self) -> FSWExplainer:
        if self._explainer is None:
            self._explainer = FSWExplainer(
                api_key=self._api_key,
                model=self._llm_model,
            )
        return self._explainer

    @torch.no_grad()
    def _infer_tensor(self, tensor: torch.Tensor) -> tuple:
        """(1,3,224,224) → (pred_idx, probs_np, branch_weights_np)"""
        logits, bw = self.model(tensor)
        probs   = torch.softmax(logits, dim=1)[0].cpu().numpy()
        weights = bw[0].cpu().numpy()
        return int(probs.argmax()), probs, weights

    def _load_image_tensor(self, source) -> tuple[torch.Tensor, Image.Image]:
        """
        Accept a file path (str/Path) or a PIL Image.
        Returns (tensor (1,3,224,224), original PIL).
        """
        if isinstance(source, (str, Path)):
            pil = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            pil = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            pil = Image.fromarray(source).convert("RGB")
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        tensor = _EVAL_TRANSFORM(pil).unsqueeze(0).to(self.device)
        return tensor, pil

    # ── Public API ────────────────────────────────────────────────────────────

    def run_single(
        self,
        image_source,
        explain: bool = True,
    ) -> dict:
        """
        Classify one image and optionally generate an LLM explanation.

        Parameters
        ----------
        image_source : file path (str/Path), PIL Image, or numpy array
        explain      : if True, call the LLM for a forensic explanation

        Returns
        -------
        dict — always contains:
            path, prediction, confidence, probs, branch_weights, risk_level
          when explain=True also contains:
            explanation (str), figures (dict of PIL Images)
        """
        t0 = time.time()
        tensor, pil = self._load_image_tensor(image_source)
        pred, probs, weights = self._infer_tensor(tensor)

        bw_dict = {
            ["Spatial (EfficientNet)", "Frequency (FFT)", "Watermark (SRM)"][i]: float(w)
            for i, w in enumerate(weights)
        }

        from explainability import _CLASS_RISK
        result = {
            "path":           str(image_source) if isinstance(image_source, (str, Path)) else "PIL_input",
            "type":           "image",
            "prediction":     self.class_names[pred],
            "confidence":     float(probs[pred]),
            "probs":          {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
            "branch_weights": bw_dict,
            "risk_level":     _CLASS_RISK.get(self.class_names[pred], "UNKNOWN"),
            "inference_ms":   round((time.time() - t0) * 1000, 1),
        }

        if explain:
            expl = self._get_explainer().explain(
                self.model, tensor, pred, probs, weights, self.class_names
            )
            result["explanation"] = expl["explanation"]
            result["figures"]     = expl["figures"]

        return result

    # -------------------------------------------------------------------------

    def run_batch(
        self,
        folder_path: str | Path,
        explain: bool = True,
        max_explain: int = 5,
        extensions: set | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """
        Classify all images in a folder.

        Parameters
        ----------
        folder_path     : directory containing images
        explain         : generate LLM explanation for up to max_explain images
        max_explain     : limit LLM calls to the N most uncertain predictions
        extensions      : set of file extensions to include (default: all images)
        progress_callback : fn(done, total, filename) called after each image

        Returns
        -------
        dict with keys:
            results   : list of per-image result dicts
            summary   : dict — counts, accuracy by class, etc.
        """
        exts   = extensions or SUPPORTED_IMAGES
        folder = Path(folder_path)
        files  = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)

        if not files:
            raise FileNotFoundError(f"No supported images found in {folder_path}")

        results = []
        for i, fpath in enumerate(files):
            if progress_callback:
                progress_callback(i, len(files), fpath.name)
            try:
                r = self.run_single(fpath, explain=False)   # skip LLM for speed
                results.append(r)
            except Exception as exc:
                results.append({"path": str(fpath), "error": str(exc), "type": "image"})

        # Explain the N most uncertain (lowest max-prob) successful results
        if explain and max_explain > 0:
            successful = [r for r in results if "error" not in r]
            uncertain  = sorted(successful, key=lambda r: r["confidence"])[:max_explain]
            for r in uncertain:
                try:
                    full = self.run_single(r["path"], explain=True)
                    r["explanation"] = full["explanation"]
                    r["figures"]     = full["figures"]
                except Exception as exc:
                    r["explanation_error"] = str(exc)

        # Summary
        preds   = [r["prediction"] for r in results if "error" not in r]
        summary = {
            "total":     len(files),
            "succeeded": len(preds),
            "failed":    len(files) - len(preds),
            "counts":    dict(Counter(preds)),
            "ai_risk_count": sum(
                1 for p in preds if p in ("ai_generated", "deepfake")
            ),
        }
        if progress_callback:
            progress_callback(len(files), len(files), "done")

        return {"results": results, "summary": summary}

    # -------------------------------------------------------------------------

    def run_video(
        self,
        video_path: str | Path,
        frame_step: int = 30,
        explain: bool = True,
    ) -> dict:
        """
        Extract frames from a video, classify each, and return an aggregated result.

        Parameters
        ----------
        video_path  : path to video file
        frame_step  : analyse every Nth frame (30 ≈ 1 frame/second at 30 fps)
        explain     : generate LLM explanation for the most representative frame

        Returns
        -------
        dict with keys:
            path, type, fps, total_frames_analyzed, overall_prediction,
            confidence, vote_distribution, frame_results,
            explanation (if explain=True), figures (if explain=True)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video inference.  "
                "Install it with:  pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_results = []
        frame_idx     = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                tens = _EVAL_TRANSFORM(pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred, probs, weights = self._infer_tensor(tens)

                frame_results.append({
                    "frame_idx":  frame_idx,
                    "timestamp":  round(frame_idx / fps, 2),
                    "prediction": self.class_names[pred],
                    "confidence": float(probs[pred]),
                    "probs":      {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
                    "_tensor":    tens,     # kept for explanation (not serialised)
                    "_pred_idx":  pred,
                    "_probs_np":  probs,
                    "_weights_np": weights,
                })

            frame_idx += 1
        cap.release()

        if not frame_results:
            raise RuntimeError("No frames could be extracted from the video.")

        # Majority vote
        all_preds    = [r["prediction"] for r in frame_results]
        vote_counter = Counter(all_preds)
        overall_pred = vote_counter.most_common(1)[0][0]

        # Best frame = highest confidence for the majority-vote class
        winner_frames = [r for r in frame_results if r["prediction"] == overall_pred]
        best_frame    = max(winner_frames, key=lambda r: r["confidence"])

        # Strip internal keys from returned frames
        public_frames = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in frame_results
        ]

        result = {
            "path":                  str(video_path),
            "type":                  "video",
            "fps":                   round(fps, 2),
            "total_frames_analyzed": len(frame_results),
            "overall_prediction":    overall_pred,
            "confidence":            best_frame["confidence"],
            "vote_distribution":     dict(vote_counter),
            "frame_results":         public_frames,
            "best_frame_timestamp":  best_frame["timestamp"],
        }

        if explain:
            expl = self._get_explainer().explain(
                self.model,
                best_frame["_tensor"],
                best_frame["_pred_idx"],
                best_frame["_probs_np"],
                best_frame["_weights_np"],
                self.class_names,
            )
            result["explanation"] = expl["explanation"]
            result["figures"]     = expl["figures"]

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers (used by the notebook input cell)
# ──────────────────────────────────────────────────────────────────────────────

def validate_path(path: str, mode: str) -> tuple[bool, str]:
    """
    Validate a user-supplied path.

    mode  : "single_image" | "batch_folder" | "single_video"

    Returns (is_valid, error_message_or_empty_string)
    """
    p = Path(path.strip())
    if not p.exists():
        return False, f"Path does not exist: {p}"

    if mode == "batch_folder":
        if not p.is_dir():
            return False, f"Expected a folder, got a file: {p}"
        imgs = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_IMAGES]
        if not imgs:
            return False, f"No supported images found in folder (need {SUPPORTED_IMAGES})"
        return True, ""

    if mode == "single_image":
        if p.is_dir():
            return False, "Expected an image file, got a folder."
        if p.suffix.lower() not in SUPPORTED_IMAGES:
            return False, f"Unsupported image format '{p.suffix}'.  Supported: {SUPPORTED_IMAGES}"
        return True, ""

    if mode == "single_video":
        if p.is_dir():
            return False, "Expected a video file, got a folder."
        if p.suffix.lower() not in SUPPORTED_VIDEOS:
            return False, f"Unsupported video format '{p.suffix}'.  Supported: {SUPPORTED_VIDEOS}"
        return True, ""

    return False, f"Unknown mode: {mode}"
