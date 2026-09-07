"""
vlm_reasoner.py
===============

The reasoning layer: a vision-language model conditioned on FSV-Net's evidence
pack, producing grounded natural-language forensic analysis.

Naming, honestly
----------------
This is **Skyra-style**, not Skyra. Skyra (arXiv 2512.15693) is a Qwen2.5-VL-7B
fine-tuned with GRPO on ViF-CoT-4K, a human-annotated artefact dataset; its
weights are not released and we have neither that dataset nor the compute. What
is reproduced here is the *idea* — grounded artefact reasoning instead of a bare
probability — on base Qwen2.5-VL, with one addition Skyra does not have: the
model is conditioned on percentile-calibrated forensic measurements, not only
on pixels. Say exactly that in the report.

The fallback ladder (this is the "CPU safety net")
--------------------------------------------------
    qwen_4bit      Qwen2.5-VL-3B in NF4 — ~3.5 GB, fits an 8 GB card
    qwen_fp16      full precision on GPU — needs ~8 GB for the 3B, more for 7B
    qwen_cpu       same weights on CPU — works, ~1-3 min per video
    anthropic_api  Claude via the API — needs ANTHROPIC_API_KEY, no local VRAM
    template       deterministic rendering of the evidence pack — always works

Every rung is tried in order and the first that loads wins. Nothing in the
pipeline hard-depends on a VLM being available: `explain()` always returns
something, and the returned dict records which backend produced it so a results
table can never silently mix them.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .evidence import (SYSTEM_PROMPT, EvidencePack, build_prompt,
                       template_explanation)
from .utils import log


# ==========================================================================
# helpers
# ==========================================================================

def _to_pil(frame: np.ndarray):
    from PIL import Image
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255 if frame.max() <= 1.5 else frame,
                        0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def _pil_to_b64(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def select_frames(frames: np.ndarray, pack: EvidencePack, k: int = 4) -> List[int]:
    """
    Which frames to show the VLM: the highest-attention frames, plus the first
    and last for temporal context, deduplicated and sorted.
    """
    T = len(frames)
    if T == 0:
        return []
    idx = set(int(np.clip(i, 0, T - 1)) for i in pack.top_frames[:max(1, k - 2)])
    idx.add(0)
    idx.add(T - 1)
    return sorted(idx)[:k]


# ==========================================================================
# Backends
# ==========================================================================

class _QwenBackend:
    def __init__(self, model_id: str, mode: str, max_new_tokens: int = 512,
                 temperature: float = 0.2):
        import torch
        from transformers import AutoProcessor

        self.mode = mode
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        kwargs: Dict[str, object] = {}
        if mode == "qwen_4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        elif mode == "qwen_fp16":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        elif mode == "qwen_cpu":
            kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = "cpu"
        else:
            raise ValueError(mode)

        # Qwen2.5-VL needs a recent transformers; the 2-VL class is the
        # fallback for older installs.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as Cls
            log.warning("Qwen2.5-VL class not found in this transformers "
                        "version; using the Qwen2-VL class. "
                        "pip install -U transformers if the load fails.")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Cls.from_pretrained(model_id, **kwargs)
        self.model.eval()
        log.info(f"VLM loaded: {model_id} [{mode}]")

    # ------------------------------------------------------------------
    def generate(self, system: str, user: str, images: List) -> str:
        import torch

        content: List[dict] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt",
                                padding=True)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=0.9)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()


class _AnthropicBackend:
    mode = "anthropic_api"

    def __init__(self, model: str, api_key_env: str, max_tokens: int = 700):
        import anthropic
        key = os.environ.get(api_key_env, "")
        if not key:
            raise RuntimeError(f"{api_key_env} not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        log.info(f"VLM backend: Anthropic API ({model})")

    def generate(self, system: str, user: str, images: List) -> str:
        content: List[dict] = []
        for im in images:
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/png",
                                       "data": _pil_to_b64(im)}})
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()


class _TemplateBackend:
    mode = "template"

    def generate(self, system: str, user: str, images: List) -> str:  # noqa: ARG002
        raise NotImplementedError   # handled specially in EvidenceReasoner


# ==========================================================================
# Public API
# ==========================================================================

@dataclass
class Explanation:
    text: str
    backend: str
    prompt: str
    n_images: int

    def to_dict(self) -> dict:
        return {"text": self.text, "backend": self.backend,
                "n_images": self.n_images}


class EvidenceReasoner:
    """
    Lazily loads the best available backend and explains evidence packs.

        reasoner = EvidenceReasoner(cfg)
        expl = reasoner.explain(pack, frames_uint8)
        print(expl.backend, expl.text)

    The backend is loaded on first use, so constructing this object is free —
    important on an 8 GB card where you do not want a VLM resident during
    training.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = None
        self._tried: List[str] = []

    # ------------------------------------------------------------------
    def load(self, force: Optional[str] = None):
        if self.backend is not None and force is None:
            return self.backend

        chain = [force] if force else list(self.cfg.vlm.fallback_chain)
        if not self.cfg.vlm.enabled:
            chain = ["template"]

        for mode in chain:
            self._tried.append(mode)
            try:
                if mode.startswith("qwen"):
                    if mode == "qwen_4bit" and not self.cfg.vlm.load_in_4bit:
                        continue
                    self.backend = _QwenBackend(
                        self.cfg.vlm.model_id, mode,
                        self.cfg.vlm.max_new_tokens, self.cfg.vlm.temperature)
                elif mode == "anthropic_api":
                    self.backend = _AnthropicBackend(
                        self.cfg.vlm.anthropic_model,
                        self.cfg.vlm.anthropic_api_key_env)
                elif mode == "template":
                    self.backend = _TemplateBackend()
                    log.info("VLM backend: deterministic template "
                             "(no model loaded — explanations will be terse "
                             "but cannot hallucinate)")
                else:
                    continue
                return self.backend
            except Exception as e:
                log.warning(f"VLM backend '{mode}' unavailable: "
                            f"{e.__class__.__name__}: {e}")

        self.backend = _TemplateBackend()
        return self.backend

    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Free the VLM's VRAM — call before training resumes."""
        from .utils import free_cuda
        self.backend = None
        free_cuda()

    # ------------------------------------------------------------------
    def explain(self, pack: EvidencePack, frames: Optional[np.ndarray] = None,
                n_images: Optional[int] = None) -> Explanation:
        backend = self.load()
        prompt = build_prompt(pack)

        if isinstance(backend, _TemplateBackend):
            return Explanation(template_explanation(pack), "template", prompt, 0)

        images = []
        if frames is not None and len(frames):
            k = n_images or self.cfg.vlm.num_evidence_frames
            images = [_to_pil(frames[i]) for i in select_frames(frames, pack, k)]

        try:
            text = backend.generate(SYSTEM_PROMPT, prompt, images)
            mode = getattr(backend, "mode", "vlm")
            return Explanation(text, mode, prompt, len(images))
        except Exception as e:
            log.warning(f"VLM generation failed ({e}) — falling back to template")
            return Explanation(template_explanation(pack), "template", prompt, 0)

    # ------------------------------------------------------------------
    def explain_batch(self, packs: Sequence[EvidencePack],
                      frames_list: Optional[Sequence[np.ndarray]] = None
                      ) -> List[Explanation]:
        out = []
        for i, p in enumerate(packs):
            fr = frames_list[i] if frames_list is not None else None
            out.append(self.explain(p, fr))
        return out


# ==========================================================================
# Evaluating the reasoning layer
# ==========================================================================

def groundedness_check(explanation: str, pack: EvidencePack) -> Dict[str, object]:
    """
    A cheap, automatic sanity check on a generated explanation: does it mention
    quantities that are actually in the evidence pack, and does it avoid
    claiming certainty on a borderline score?

    This is not a substitute for human evaluation of reasoning quality (that
    needs annotations you do not have). It is a guard-rail that catches the
    obvious failure — a fluent paragraph about "unnatural blinking" when the
    detector measured nothing of the sort.
    """
    text = explanation.lower()
    mentioned = [k for k in pack.descriptors if k.replace("_", " ") in text
                 or k in text]
    branch_mentions = [b for b in pack.branch_attention if b in text]

    hedges = ["inconclusive", "borderline", "uncertain", "not conclusive",
              "cannot conclude", "weak evidence"]
    hedged = any(h in text for h in hedges)

    unsupported_terms = ["blink", "eye movement", "lip sync", "lip-sync",
                         "pulse", "heartbeat", "shadow", "reflection", "teeth"]
    unsupported = [t for t in unsupported_terms if t in text]

    return {
        "descriptors_mentioned": mentioned,
        "branches_mentioned": branch_mentions,
        "hedged": hedged,
        "should_hedge": pack.confidence_band in ("borderline", "weak"),
        "hedging_correct": hedged or pack.confidence_band not in ("borderline", "weak"),
        "ungrounded_claims": unsupported,
        "grounded_score": round(
            (len(mentioned) > 0) * 0.4
            + (len(branch_mentions) > 0) * 0.2
            + (not unsupported) * 0.2
            + (hedged or pack.confidence_band not in ("borderline", "weak")) * 0.2, 3),
    }


__all__ = ["EvidenceReasoner", "Explanation", "groundedness_check",
           "select_frames"]
