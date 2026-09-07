"""
qwen_detector.py
================

**Qwen2.5-VL as the detector itself**, QLoRA fine-tuned — the option you chose.

This is a second, independent detector path. `model.FSVNet` still exists and
still works; this one swaps the whole backbone for a vision-language model and
keeps TSC as an *input* rather than a branch.

The integration, and why it is not just "Skyra again"
------------------------------------------------------
Skyra hands its VLM frames and asks it to reason about artefacts. This one
hands the VLM **frames *and* the percentile-calibrated forensic measurements**
TSC produces, and fine-tunes it to answer with a verdict:

    <image> <image> <image> <image>
    Forensic measurements for this clip (percentile = position among real videos):
      temporal_whiteness   0.412   (3rd pct of real)   <-- ANOMALOUS
      temporal_peakiness   0.310   (97th pct of real)  <-- ANOMALOUS
      ...
    Is this video real or manipulated? Answer with one word.
    Answer: fake

A VLM cannot see a temporal spectral anomaly — no amount of looking at frames
reveals it, because it lives in the relationship *between* frames at a
frequency scale the tokeniser never represents. Giving it the measurement is
the only way it can use that evidence. That makes the ablation clean and the
contribution precise:

    Qwen + frames                    <- the Skyra-style baseline
    Qwen + frames + TSC measurements <- this work
    FSV-Net (no VLM)                 <- the cheap model

and the difference between rows 1 and 2 is exactly the value of the forensic
channel, measured rather than asserted.

How the verdict is read
-----------------------
Not by generating text and string-matching it — that is unstable and gives no
probability. Instead **constrained-token scoring**: the prompt ends at
`Answer:` and we read the logits of the two answer tokens (` real`, ` fake`)
at that position, then softmax over just those two. That yields a calibrated
`p(fake)` you can put on an ROC curve, it is differentiable so it is also the
training objective, and the model stays generative so it can still write the
explanation afterwards.

Honest status
-------------
Everything here that does not need model weights is unit-tested
(`scripts/test_qwen_path.py`). The parts that need Qwen weights are **not**,
because this environment has no access to Hugging Face — the first real run
happens on your machine. `preflight()` exists for exactly that reason: run it
before you start a training job, not after.

8 GB is tight but workable
--------------------------
3B in NF4 ≈ 2.2 GB of weights. The vision tower stays **frozen** (LoRA on the
language model only) — unfreezing it is what turns "tight" into OOM. With 4
frames at `max_pixels = 200704`, gradient checkpointing, batch 1 and
accumulation 16, expect ~6-7 GB and roughly 2-4 s/step on a 4060. A 2,000-step
run is therefore a couple of hours. If you OOM, `preflight()` prints the knobs
in the order to turn them.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import log, save_json


# ==========================================================================
# Config
# ==========================================================================

@dataclass
class QwenDetectorConfig:
    # ------------------------------------------------------------------
    # WHICH CHECKPOINT — read this before you write a results table
    # ------------------------------------------------------------------
    # "Qwen/Qwen2.5-VL-3B-Instruct"  base VLM, no forensic training. This is
    #                                what a 4060 8 GB can actually FINE-TUNE.
    # "Qwen/Qwen2.5-VL-7B-Instruct"  base VLM at Skyra's size. NF4 inference
    #                                only on 8 GB; cannot be fine-tuned there.
    # SKYRA_MODEL_ID                 the released Skyra weights (CVPR 2026,
    #                                arXiv 2512.15693). Inference only here.
    #
    # Only a run that loads SKYRA_MODEL_ID may be labelled "Skyra" in a table.
    # A base Qwen with a forensic prompt is "Skyra-style", and calling it
    # anything else is the fastest way to lose an examiner's trust.
    #
    # AND NOTE WHAT SKYRA IS FOR: it was trained on fully AI-GENERATED video
    # (Wan, CogVideoX, Hunyuan, Sora-2) and is never evaluated on Celeb-DF or
    # FaceForensics++. Its artifact taxonomy has no category for localised
    # face replacement. Running it on Celeb-DF-v2 is an OUT-OF-DISTRIBUTION
    # TRANSFER TEST, not a like-for-like baseline. Report it as such.
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_in_4bit: bool = True

    # ---- input ----
    num_frames: int = 4               # frames shown to the VLM per clip
    max_pixels: int = 256 * 28 * 28   # Qwen dynamic resolution cap per image
    min_pixels: int = 64 * 28 * 28
    use_tsc_evidence: bool = True     # <- the ablation switch that matters
    use_frames: bool = True           # False = measurements only (a real control)

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj")
    freeze_vision_tower: bool = True  # unfreezing this is how you OOM on 8 GB

    # ---- training ----
    epochs: int = 2
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    grad_checkpointing: bool = True
    save_every: int = 200
    log_every: int = 10
    seed: int = 42

    # ---- answer vocabulary ----
    answer_real: str = " real"
    answer_fake: str = " fake"

    # ---- evidence channels (the ablation switches) ----
    use_lmf_evidence: bool = True     # landmark motion-compensated descriptors

    def is_official_skyra(self) -> bool:
        """True only when the released Skyra weights are being loaded."""
        return "skyra" in str(self.model_id).lower()

    def condition_name(self) -> str:
        """The label this configuration is allowed to carry in a results table."""
        if self.is_official_skyra():
            return "Skyra (official weights, zero-shot OOD)"
        base = str(self.model_id).split("/")[-1]
        ch = [c for c, on in [("frames", self.use_frames),
                              ("TSC", self.use_tsc_evidence),
                              ("LMF", self.use_lmf_evidence)] if on]
        return f"{base} + " + "+".join(ch) if ch else base


# The released Skyra checkpoint. Set this from the model card at
# https://github.com/JoeLeelyf/Skyra (HuggingFace collection) before running
# the Skyra condition; it is deliberately NOT guessed here, because a wrong
# repo id that silently falls back to base Qwen would produce a table row
# labelled "Skyra" that is not Skyra.
SKYRA_MODEL_ID: str = ""


def resolve_skyra_id(explicit: str = "") -> str:
    """
    Returns the Skyra repo id from, in order: the argument, $SKYRA_MODEL_ID,
    the module constant. Raises if none is set — refusing to run is correct
    here, because the alternative is mislabelled results.
    """
    import os
    for cand in (explicit, os.environ.get("SKYRA_MODEL_ID", ""), SKYRA_MODEL_ID):
        if cand:
            return cand
    raise RuntimeError(
        "No Skyra checkpoint configured. Set SKYRA_MODEL_ID (env var or "
        "fsvnet.qwen_detector.SKYRA_MODEL_ID) to the repo id from the "
        "HuggingFace collection linked at https://github.com/JoeLeelyf/Skyra .\n"
        "Until then the Skyra row must be reported as NOT RUN — do not "
        "substitute base Qwen and call it Skyra.")


# ==========================================================================
# Preflight
# ==========================================================================

def preflight(cfg: QwenDetectorConfig, verbose: bool = True) -> Dict[str, object]:
    """
    Check every dependency and estimate memory BEFORE a training run starts.
    Returns a report; never raises.
    """
    rep: Dict[str, object] = {"ok": True, "problems": [], "warnings": [], "info": {}}

    def need(mod: str, hint: str):
        try:
            m = __import__(mod)
            rep["info"][mod] = getattr(m, "__version__", "?")
        except ImportError:
            rep["ok"] = False
            rep["problems"].append(f"{mod} missing — {hint}")

    need("torch", "pip install torch")
    need("transformers", "pip install -U 'transformers>=4.49'")
    need("peft", "pip install peft")
    need("accelerate", "pip install accelerate")
    if cfg.load_in_4bit:
        need("bitsandbytes", "pip install bitsandbytes  (4-bit needs it)")
    try:
        __import__("qwen_vl_utils")
        rep["info"]["qwen_vl_utils"] = "present"
    except ImportError:
        rep["warnings"].append("qwen-vl-utils not installed — falling back to "
                               "the processor's own image handling (fine, but "
                               "pip install qwen-vl-utils is the tested path)")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        rep["info"]["qwen2_5_vl_class"] = "available"
    except Exception:
        rep["warnings"].append(
            "transformers has no Qwen2_5_VLForConditionalGeneration — your "
            "version predates Qwen2.5-VL. pip install -U transformers")

    # ---- memory ----
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            rep["info"]["gpu"] = torch.cuda.get_device_name(0)
            rep["info"]["vram_gb"] = round(vram, 1)
            billions = 7 if "7B" in cfg.model_id else 3
            weights = billions * (0.55 if cfg.load_in_4bit else 2.0)
            acts = 0.35 * cfg.num_frames * (cfg.max_pixels / (256 * 28 * 28))
            est = weights + acts + 1.2
            rep["info"]["estimated_peak_gb"] = round(est, 1)
            if est > vram * 0.92:
                rep["ok"] = False
                rep["problems"].append(
                    f"estimated peak {est:.1f} GB vs {vram:.1f} GB available")
        else:
            rep["warnings"].append("no CUDA — a VLM on CPU is minutes per step; "
                                   "usable for a smoke test, not for training")
    except Exception as e:
        rep["warnings"].append(f"could not probe the GPU: {e}")

    if verbose:
        print("=" * 72)
        print("QWEN DETECTOR PREFLIGHT")
        print("=" * 72)
        for k, v in rep["info"].items():
            print(f"  {k:<22} {v}")
        for w in rep["warnings"]:
            print(f"  WARN    {w}")
        for p in rep["problems"]:
            print(f"  BLOCKER {p}")
        if not rep["ok"]:
            print("\nIf memory is the blocker, turn these in order:")
            print("  1. cfg.num_frames = 2")
            print("  2. cfg.max_pixels = 128 * 28 * 28")
            print("  3. cfg.lora_r = 8")
            print("  4. cfg.model_id = 'Qwen/Qwen2.5-VL-3B-Instruct' (if on 7B)")
            print("  5. cfg.use_frames = False  — measurements-only; tiny, and a")
            print("     legitimate ablation row rather than a defeat")
        else:
            print("\n  READY")
        print("=" * 72)
    return rep


# ==========================================================================
# Prompt construction  (pure, unit-testable, no weights needed)
# ==========================================================================

SYSTEM = ("You are a video forensics model. You judge whether a video clip is "
          "authentic camera footage or synthetically generated or manipulated. "
          "You answer with exactly one word.")

QUESTION = "Is this video real or manipulated? Answer with one word."


def format_evidence(descriptors: Dict[str, float],
                    percentiles: Optional[Dict[str, Optional[float]]] = None,
                    context_only: Optional[Sequence[str]] = None) -> str:
    """
    Render the forensic measurements as the text block the VLM reads.

    Percentiles are what make this usable: a raw float means nothing to a
    language model, while "3rd percentile of real videos" is a statement it can
    reason with and a reader can check.
    """
    from .evidence import CONTEXT_ONLY, _ordinal

    context_only = set(context_only or CONTEXT_ONLY)
    lines = ["Forensic measurements for this clip "
             "(percentile = this clip's position among known-real videos):"]
    for k, v in descriptors.items():
        p = (percentiles or {}).get(k)
        if p is None:
            tag = "not calibrated"
        else:
            tag = f"{_ordinal(p)} pct of real"
        flag = ""
        if k in context_only:
            flag = "  (context only)"
        elif p is not None and (p <= 5 or p >= 95):
            flag = "  <-- ANOMALOUS"
        lines.append(f"  {k:<26} {v:+.4f}   ({tag}){flag}")
    return "\n".join(lines)


def build_messages(n_images: int, evidence_text: Optional[str],
                   answer: Optional[str] = None) -> List[dict]:
    """
    The chat structure. `answer` is set during training (teacher forcing) and
    left None at inference, where the model is scored at the answer position.
    """
    content: List[dict] = [{"type": "image"} for _ in range(n_images)]
    text = ""
    if evidence_text:
        text += evidence_text + "\n\n"
    text += QUESTION
    content.append({"type": "text", "text": text})

    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": answer}]})
    return msgs


def select_frames(n_available: int, k: int) -> List[int]:
    """Evenly spaced frame indices — deterministic, so eval is reproducible."""
    if n_available <= k:
        return list(range(n_available))
    return [int(round(i * (n_available - 1) / (k - 1))) for i in range(k)]


# ==========================================================================
# The detector
# ==========================================================================

class QwenVLDetector:
    """
    Wraps Qwen2.5-VL + LoRA adapters and exposes `score_clip` / `train_on`.

    Construct with `load=False` to build the object without touching the
    weights — useful for testing prompt construction, and for `preflight`.
    """

    def __init__(self, cfg: QwenDetectorConfig, load: bool = True,
                 adapter_path: Optional[str] = None):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.answer_ids: Tuple[int, int] = (-1, -1)
        if load:
            self.load(adapter_path)

    # ------------------------------------------------------------------
    def load(self, adapter_path: Optional[str] = None, for_training: bool = False):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        cfg = self.cfg
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as Cls
            log.warning("Qwen2.5-VL class unavailable; using the Qwen2-VL class")

        kwargs: Dict[str, object] = {"device_map": "auto"}
        if cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = Cls.from_pretrained(cfg.model_id, **kwargs)

        self._resolve_answer_tokens()

        if for_training:
            self._attach_lora()
        elif adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            log.info(f"LoRA adapters loaded from {adapter_path}")

        self.model.eval()
        log.info(f"Qwen detector ready: {cfg.model_id} "
                 f"(4bit={cfg.load_in_4bit}, frames={cfg.num_frames})")
        return self

    # ------------------------------------------------------------------
    def _resolve_answer_tokens(self) -> None:
        """
        Find the single token id for each answer word.

        If a word tokenises to more than one token we take the first and say so
        — scoring the first token is still a valid two-way decision as long as
        the two first-tokens differ, and the assertion below is what catches
        the case where they do not.
        """
        cfg = self.cfg
        ids = []
        for word in (cfg.answer_real, cfg.answer_fake):
            toks = self.tokenizer.encode(word, add_special_tokens=False)
            if len(toks) != 1:
                log.warning(f"answer word {word!r} is {len(toks)} tokens "
                            f"{toks}; scoring its first token only")
            ids.append(toks[0])
        if ids[0] == ids[1]:
            raise ValueError(
                f"answer words {cfg.answer_real!r} and {cfg.answer_fake!r} share "
                "a first token — pick different words (e.g. ' authentic' / "
                "' synthetic')")
        self.answer_ids = (ids[0], ids[1])
        log.info(f"answer token ids: real={ids[0]}, fake={ids[1]}")

    # ------------------------------------------------------------------
    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        cfg = self.cfg
        if cfg.load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=cfg.grad_checkpointing)
        if cfg.grad_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                        lora_dropout=cfg.lora_dropout,
                        target_modules=list(cfg.lora_targets),
                        bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lc)

        if cfg.freeze_vision_tower:
            frozen = 0
            for name, p in self.model.named_parameters():
                if ("visual" in name or "vision" in name) and p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
            log.info(f"vision tower frozen ({frozen} tensors) — this is what "
                     "keeps the run inside 8 GB")
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    def _prepare(self, frames_u8: Optional[np.ndarray], evidence_text: Optional[str],
                 answer: Optional[str] = None):
        """Build the model inputs for one clip."""
        from PIL import Image

        cfg = self.cfg
        images: List = []
        if cfg.use_frames and frames_u8 is not None and len(frames_u8):
            idx = select_frames(len(frames_u8), cfg.num_frames)
            images = [Image.fromarray(frames_u8[i]) for i in idx]

        msgs = build_messages(len(images), evidence_text, answer)
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=(answer is None))
        inputs = self.processor(text=[text], images=images or None,
                                return_tensors="pt", padding=True)
        return {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    # ------------------------------------------------------------------
    def score_clip(self, frames_u8: Optional[np.ndarray],
                   descriptors: Optional[Dict[str, float]] = None,
                   percentiles: Optional[Dict[str, Optional[float]]] = None
                   ) -> Dict[str, float]:
        """
        Return {"p_fake": float, "logit_real": float, "logit_fake": float}.

        Constrained-token scoring: softmax over just the two answer tokens at
        the first generated position.
        """
        import torch

        ev = (format_evidence(descriptors or {}, percentiles)
              if (self.cfg.use_tsc_evidence and descriptors) else None)
        inputs = self._prepare(frames_u8, ev, answer=None)

        with torch.no_grad():
            out = self.model(**inputs)
        last = out.logits[0, -1].float()          # next-token distribution
        lr, lf = last[self.answer_ids[0]], last[self.answer_ids[1]]
        p_fake = float(torch.softmax(torch.stack([lr, lf]), dim=0)[1])
        return {"p_fake": p_fake, "logit_real": float(lr), "logit_fake": float(lf)}

    # ------------------------------------------------------------------
    def explain_clip(self, frames_u8: Optional[np.ndarray],
                     descriptors: Optional[Dict[str, float]] = None,
                     percentiles: Optional[Dict[str, Optional[float]]] = None,
                     max_new_tokens: int = 256) -> str:
        """Free-text rationale from the same fine-tuned model."""
        import torch

        ev = (format_evidence(descriptors or {}, percentiles)
              if (self.cfg.use_tsc_evidence and descriptors) else None)
        msgs = build_messages(
            self.cfg.num_frames if (self.cfg.use_frames and frames_u8 is not None) else 0,
            ev, None)
        msgs[1]["content"][-1]["text"] = (
            (ev + "\n\n" if ev else "")
            + "Explain, in at most 120 words, what the evidence above indicates "
              "about whether this video is authentic. Ground every claim in a "
              "listed measurement or something visible in the frames.")
        from PIL import Image
        images = []
        if self.cfg.use_frames and frames_u8 is not None and len(frames_u8):
            idx = select_frames(len(frames_u8), self.cfg.num_frames)
            images = [Image.fromarray(frames_u8[i]) for i in idx]

        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None,
                                return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------
    def _loss_for_example(self, frames_u8, descriptors, percentiles, label: int):
        """
        Cross-entropy on the two answer tokens only.

        Full LM loss over the whole sequence would spend most of its gradient
        on reproducing the prompt, which is neither what we want to learn nor
        what we measure at eval. Restricting it to the answer position makes
        training and evaluation the same objective.
        """
        import torch
        import torch.nn.functional as F

        ev = (format_evidence(descriptors or {}, percentiles)
              if (self.cfg.use_tsc_evidence and descriptors) else None)
        inputs = self._prepare(frames_u8, ev, answer=None)
        out = self.model(**inputs)
        last = out.logits[0, -1].float()
        pair = torch.stack([last[self.answer_ids[0]], last[self.answer_ids[1]]])
        target = torch.tensor(int(label), device=pair.device)
        return F.cross_entropy(pair.unsqueeze(0), target.unsqueeze(0)), pair

    # ------------------------------------------------------------------
    def train_on(self, examples: Sequence[dict], out_dir: str,
                 val_examples: Optional[Sequence[dict]] = None) -> Dict[str, object]:
        """
        QLoRA fine-tune.

        examples: [{"frames": (T,H,W,3) uint8 | None,
                    "descriptors": {...}, "percentiles": {...},
                    "label": 0|1, "video": str}, ...]

        Build them with `build_examples()` below.
        """
        import torch
        from torch.optim import AdamW

        cfg = self.cfg
        if self.model is None:
            self.load(for_training=True)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable parameters — LoRA was not attached; "
                               "call load(for_training=True)")
        opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        steps_total = max(1, (len(examples) * cfg.epochs) // cfg.grad_accum)
        warm = max(1, int(steps_total * cfg.warmup_ratio))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))

        rng = np.random.default_rng(cfg.seed)
        history: List[dict] = []
        self.model.train()
        gstep = 0
        t0 = time.time()

        for epoch in range(cfg.epochs):
            order = rng.permutation(len(examples))
            running, n = 0.0, 0
            for i, idx in enumerate(order):
                ex = examples[int(idx)]
                loss, _ = self._loss_for_example(
                    ex.get("frames"), ex.get("descriptors"),
                    ex.get("percentiles"), ex["label"])
                (loss / cfg.grad_accum).backward()
                running += float(loss); n += 1

                if (i + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    gstep += 1
                    if gstep % cfg.log_every == 0:
                        log.info(f"epoch {epoch+1} step {gstep}/{steps_total} "
                                 f"loss {running/max(1,n):.4f} "
                                 f"lr {sched.get_last_lr()[0]:.2e} "
                                 f"({time.time()-t0:.0f}s)")
                        running, n = 0.0, 0
                    if cfg.save_every and gstep % cfg.save_every == 0:
                        self.save(out_dir / f"step{gstep}")

            rec = {"epoch": epoch + 1, "steps": gstep}
            if val_examples:
                rec.update(self.evaluate(val_examples, split="val"))
                self.model.train()
            history.append(rec)

        self.save(out_dir / "final")
        save_json({"history": history, "config": cfg.__dict__},
                  out_dir / "train_history.json")
        return {"history": history, "adapter_dir": str(out_dir / "final")}

    # ------------------------------------------------------------------
    def evaluate(self, examples: Sequence[dict], split: str = "test"
                 ) -> Dict[str, float]:
        """Video-level metrics, identical to the FSV-Net evaluation path."""
        from .evaluate import full_metrics

        self.model.eval()
        per_video: Dict[str, List[float]] = {}
        labels: Dict[str, int] = {}
        for ex in examples:
            s = self.score_clip(ex.get("frames"), ex.get("descriptors"),
                                ex.get("percentiles"))
            v = ex.get("video", str(id(ex)))
            per_video.setdefault(v, []).append(s["p_fake"])
            labels[v] = int(ex["label"])

        vids = sorted(per_video)
        y = np.array([labels[v] for v in vids])
        p = np.array([float(np.mean(per_video[v])) for v in vids])
        m = full_metrics(y, p)
        log.info(f"[qwen:{split}] AUC {m['auc']:.4f}  acc {m['acc']:.4f}  n={m['n']}")
        return {f"{split}_{k}": v for k, v in m.items()
                if isinstance(v, (int, float))}

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        (path / "qwen_detector_config.json").write_text(
            json.dumps(self.cfg.__dict__, indent=2, default=str))
        log.info(f"adapters saved: {path}")


# ==========================================================================
# Turning the FSV-Net dataset into VLM examples
# ==========================================================================

def build_examples(fsv_model, loader, cfg, reference=None,
                   qcfg: Optional[QwenDetectorConfig] = None,
                   max_items: Optional[int] = None) -> List[dict]:
    """
    Run FSV-Net over a loader to harvest the TSC measurements, and pair them
    with the frames — producing the example list `train_on` expects.

    Note what this implies: the forensic branch is computed **once, offline**,
    and the VLM never has to learn spectral analysis. That is the whole point
    of the integration — the measurement is cheap and exact, and the VLM's job
    is to weigh it against what it sees.
    """
    import torch

    from .explain import denormalise

    qcfg = qcfg or QwenDetectorConfig()
    fsv_model.eval()
    out: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            clip = batch["clip"].to(cfg.device)
            mask = batch["mask"].to(cfg.device)
            _, info = fsv_model(clip, mask, return_evidence=True)
            desc_t = info.get("evidence", {}).get("tsc_descriptors", {})

            for b in range(clip.shape[0]):
                descriptors = {k: float(v[b]) for k, v in desc_t.items()}
                percentiles = ({k: reference.percentile_of(k, v)
                                for k, v in descriptors.items()}
                               if reference is not None else {})
                out.append({
                    "frames": denormalise(clip[b].cpu()),
                    "descriptors": descriptors,
                    "percentiles": percentiles,
                    "label": int(batch["label"][b]),
                    "video": f"v{int(batch['video_index'][b])}",
                })
                if max_items and len(out) >= max_items:
                    log.info(f"built {len(out)} VLM examples (capped)")
                    return out
    log.info(f"built {len(out)} VLM examples")
    return out


# ==========================================================================
# The ablation that measures the contribution
# ==========================================================================

QWEN_ABLATIONS = {
    "qwen_frames_only":      dict(use_frames=True,  use_tsc_evidence=False),
    "qwen_frames_plus_tsc":  dict(use_frames=True,  use_tsc_evidence=True),
    "qwen_tsc_only":         dict(use_frames=False, use_tsc_evidence=True),
}


def run_qwen_ablation(train_ex, val_ex, test_ex, out_root,
                      base: Optional[QwenDetectorConfig] = None,
                      variants: Optional[Dict[str, dict]] = None
                      ) -> "object":
    """
    Train one adapter per configuration and tabulate test AUC.

    `qwen_frames_only` is the Skyra-style baseline; `qwen_frames_plus_tsc` is
    this work. The difference between them, with a confidence interval, is the
    contribution — stated as a measurement rather than a claim.
    """
    import copy

    import pandas as pd

    base = base or QwenDetectorConfig()
    variants = variants or QWEN_ABLATIONS
    rows = []
    for name, flags in variants.items():
        log.info("=" * 70)
        log.info(f"QWEN ABLATION: {name}  {flags}")
        log.info("=" * 70)
        c = copy.deepcopy(base)
        for k, v in flags.items():
            setattr(c, k, v)
        det = QwenVLDetector(c, load=False)
        det.load(for_training=True)
        det.train_on(train_ex, Path(out_root) / name, val_ex)
        m = det.evaluate(test_ex, split="test")
        rows.append({"variant": name, **flags,
                     **{k.replace("test_", ""): v for k, v in m.items()}})
        del det
        from .utils import free_cuda
        free_cuda()

    table = pd.DataFrame(rows)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_root) / "qwen_ablation.csv", index=False)
    print(table.to_string(index=False))
    return table


__all__ = ["QwenDetectorConfig", "QwenVLDetector", "preflight", "build_examples",
           "format_evidence", "build_messages", "select_frames",
           "run_qwen_ablation", "QWEN_ABLATIONS"]
