"""
inference.py
============

End-to-end single-video inference: path in, verdict + evidence + explanation
out. Deliberately UI-agnostic so a Gradio/Streamlit/FastAPI front end is a
thin wrapper, exactly as `../DeepfakeProject/src/inference_pipeline.py` was for
the image model.

    from fsvnet.inference import VideoInferencePipeline
    pipe = VideoInferencePipeline(cfg, "checkpoints/fsvnet_v1_best.pt")
    r = pipe.run("clip.mp4", explain=True)
    print(r["verdict"], r["fake_probability"])
    print(r["explanation"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .dataset import normalise
from .evidence import EvidencePack, ReferenceStats, build_evidence
from .faces import build_detector, crop_clip
from .utils import log
from .video_io import read_clip
from .vlm_reasoner import EvidenceReasoner


class VideoInferencePipeline:
    def __init__(self, cfg, checkpoint: str,
                 reference_stats: Optional[str] = None,
                 load_vlm_lazily: bool = True):
        from .train import load_checkpoint

        self.cfg = cfg
        self.model, self.ckpt_meta = load_checkpoint(checkpoint, None, cfg.device)

        # A checkpoint carries the geometry it was trained at, and the forensic
        # branches are not scale-invariant — running a 112 px model on 224 px
        # crops changes every spectral statistic it learned. So the checkpoint
        # wins over the caller's config, loudly.
        ck_data = (self.ckpt_meta.get("config") or {}).get("data") or {}
        for field in ("img_size", "azimuthal_bins"):
            want = ck_data.get(field)
            if want is None:
                continue
            have = getattr(cfg.data, field, None)
            if have is not None and have != want:
                log.warning(f"checkpoint was trained with {field}={want} but the "
                            f"config says {have} — using {want}. Forensic "
                            "features are resolution-dependent; this is not "
                            "a preference.")
                setattr(cfg.data, field, want)
        self.detector = build_detector(cfg.data.face_backend)
        self.reference = (ReferenceStats.load(reference_stats)
                          if reference_stats and Path(reference_stats).exists()
                          else None)
        if self.reference is None:
            log.warning("No reference stats — explanations will not be "
                        "percentile-calibrated. Build them once with "
                        "evidence.build_reference_stats() on the training reals.")
        self.reasoner = EvidenceReasoner(cfg)
        if not load_vlm_lazily:
            self.reasoner.load()

    # ------------------------------------------------------------------
    def _prepare(self, video_path: str, clip_id: int = 0, n_clips: int = 1):
        frames, info = read_clip(video_path, self.cfg.data.num_frames,
                                 self.cfg.data.frame_stride,
                                 mode="grid" if n_clips > 1 else "center",
                                 clip_id=clip_id, n_clips=n_clips)
        if len(frames) == 0:
            raise RuntimeError(f"could not decode any frames from {video_path}")
        crop = crop_clip(frames, self.detector, self.cfg.data.img_size,
                         self.cfg.data.face_margin, self.cfg.data.detect_every)
        x = normalise(crop.frames).unsqueeze(0).to(self.cfg.device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.cfg.device)
        return x, mask, crop.frames, {**info, "face_ratio": crop.face_ratio}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, video_path: str, explain: bool = True,
            n_clips: Optional[int] = None, make_figures: bool = True) -> Dict:
        """
        Score one video. With n_clips>1 the clips are averaged into one
        video-level probability (the same aggregation used in evaluation), and
        the evidence pack is built from the most confident clip.
        """
        n_clips = n_clips or self.cfg.data.clips_per_video_eval
        probs: List[float] = []
        packs: List[EvidencePack] = []
        frames_list: List[np.ndarray] = []
        infos: List[dict] = []

        for c in range(max(1, n_clips)):
            x, mask, frames_u8, info = self._prepare(video_path, c, n_clips)
            pack = build_evidence(self.model, x, mask, video_path, self.cfg,
                                  self.reference, frames_u8,
                                  make_figures=(make_figures and c == 0))
            probs.append(pack.fake_probability)
            packs.append(pack)
            frames_list.append(frames_u8)
            infos.append(info)

        mean_p = float(np.mean(probs))
        # report the evidence of the clip the model was most sure about
        k = int(np.argmax([abs(p - 0.5) for p in probs]))
        pack = packs[k]

        result: Dict[str, object] = {
            "video_path": str(video_path),
            "fake_probability": mean_p,
            "verdict": "fake" if mean_p >= 0.5 else "real",
            "confidence_band": pack.confidence_band,
            "clip_probabilities": probs,
            "n_clips": len(probs),
            "face_ratio": infos[k].get("face_ratio"),
            "sample_hz": infos[k].get("sample_hz"),
            "evidence": pack.to_dict(),
            "figures": pack.figures,
            "checkpoint": self.ckpt_meta.get("epoch"),
        }

        if explain:
            expl = self.reasoner.explain(pack, frames_list[k])
            result["explanation"] = expl.text
            result["explanation_backend"] = expl.backend
            from .vlm_reasoner import groundedness_check
            result["groundedness"] = groundedness_check(expl.text, pack)

        return result

    # ------------------------------------------------------------------
    def run_folder(self, folder: str, explain: bool = False,
                   max_explain: int = 5, pattern: str = "*") -> List[Dict]:
        from .datasets_fetch import VIDEO_EXTS
        vids = [p for p in sorted(Path(folder).rglob(pattern))
                if p.suffix.lower() in VIDEO_EXTS]
        log.info(f"scoring {len(vids)} videos in {folder}")
        out = []
        for i, p in enumerate(vids):
            try:
                out.append(self.run(str(p), explain=(explain and i < max_explain),
                                    make_figures=(i < max_explain)))
            except Exception as e:
                log.warning(f"{p.name}: {e}")
                out.append({"video_path": str(p), "error": str(e)})
        return out


__all__ = ["VideoInferencePipeline"]
