"""
streamlit_app.py
================

Frontend for FSV-Net.

    cd video_detector
    streamlit run app/streamlit_app.py

Upload a video, get a verdict, and — more importantly — get the four separate
answers to *why*: where in the frame, when in the clip, which branch the
decision depended on, and which measured quantity was abnormal.

Design note: the verdict is deliberately **not** the biggest thing on the page.
A detector at 0.83 AUC is wrong about one video in six, and a UI that shows a
huge red FAKE and nothing else trains the user to stop reading. The confidence
band, the calibration caveat and the evidence share the top of the page with it.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="FSV-Net · Deepfake Video Detection",
                   page_icon="🎞️", layout="wide")

# --------------------------------------------------------------------------
# Style — matches the project's figure palette (src/fsvnet/viz.py)
# --------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1350px; }
  .verdict-card {
      border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: .4rem;
      border: 1px solid rgba(128,128,128,.25);
  }
  .verdict-fake { background: rgba(227,73,72,.10);  border-left: 5px solid #e34948; }
  .verdict-real { background: rgba(0,131,0,.10);    border-left: 5px solid #008300; }
  .verdict-unsure { background: rgba(237,161,0,.10); border-left: 5px solid #eda100; }
  .verdict-card h2 { margin: 0 0 .25rem 0; font-size: 1.5rem; }
  .verdict-card p  { margin: 0; opacity: .85; font-size: .92rem; }
  .caveat { font-size: .85rem; opacity: .75; }
  code { font-size: .85em; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(ckpt: str, ref: str, device: str, frames: int, img_size: int,
                  face_backend: str):
    from fsvnet.config import Config
    from fsvnet.inference import VideoInferencePipeline

    cfg = Config()
    cfg.device = device
    cfg.data.num_frames = frames
    cfg.data.img_size = img_size
    cfg.data.face_backend = face_backend
    cfg.data.min_faces_per_clip = 0.0        # never refuse a user's upload
    cfg.paths.mkdirs()
    pipe = VideoInferencePipeline(cfg, ckpt,
                                  reference_stats=ref or None,
                                  load_vlm_lazily=True)
    # cfg.data.img_size has now been rewritten to the checkpoint's value
    return pipe, cfg


def find_checkpoints() -> list:
    return sorted(str(p) for p in (ROOT / "checkpoints").glob("*.pt"))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("FSV-Net")
st.sidebar.caption("Frequency · Spatial · Watermark · **Temporal**")

ckpts = find_checkpoints()
ckpt = st.sidebar.selectbox(
    "Model checkpoint", ckpts or ["— none found —"],
    help="Train one with the notebook, or point this at any *_best.pt")
ckpt_ok = bool(ckpts)

ref_default = ROOT / "outputs" / "reference_stats.json"
ref = st.sidebar.text_input("Reference stats (optional)",
                            str(ref_default) if ref_default.exists() else "")
if not ref:
    st.sidebar.warning("Without reference stats the measurements cannot be "
                       "turned into percentiles, so the explanation will say "
                       "they are uncalibrated rather than guess.", icon="⚠️")

with st.sidebar.expander("Inference settings", expanded=False):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = st.selectbox("Device", ["cuda", "cpu"] if has_cuda else ["cpu"])
    frames = st.slider("Frames per clip", 8, 32, 16, step=8)
    # Crop size is NOT a user choice: the forensic branches are built around a
    # fixed radial-ring index, so a model trained at 112 px must be run at
    # 112 px. The pipeline reads the value out of the checkpoint and overrides
    # whatever is passed in — this is just the seed value.
    img_size = 224
    face_backend = st.selectbox("Face detection",
                                ["auto", "mediapipe", "opencv", "none"])
    n_clips = st.slider("Clips per video", 1, 4, 2,
                        help="Scores are averaged across clips, as in evaluation")

with st.sidebar.expander("Explainability", expanded=True):
    do_cam = st.checkbox("Grad-CAM (where)", True)
    do_occ = st.checkbox("Temporal occlusion (when)", True,
                         help="Costs one forward pass per frame")
    occ_mode = st.radio("Occlusion mode", ["freeze", "blank"], horizontal=True)
    do_reason = st.checkbox("Written reasoning", True,
                            help="Qwen2.5-VL if installed, else a deterministic "
                                 "template that cannot hallucinate")

st.sidebar.divider()
st.sidebar.caption(
    "This is a research prototype. Do not use a single score to make a claim "
    "about a real person or a real video.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("Deepfake video detection")
st.caption("Four forensic signals, fused — and four separate answers to *why*.")

if not ckpt_ok:
    st.error("No checkpoint found in `checkpoints/`. Run "
             "`notebooks/FSV_Net_RUN_ALL.ipynb` (it trains one in a few minutes) "
             "or copy a `.pt` file there.")
    st.stop()

up = st.file_uploader("Upload a video",
                      type=["mp4", "avi", "mov", "mkv", "webm", "m4v"])
col_a, col_b = st.columns([2, 1])
with col_b:
    st.markdown("**Or use a local path**")
    local = st.text_input("Path", "", label_visibility="collapsed",
                          placeholder=r"D:\data\clip.mp4")

video_path = None
if up is not None:
    tmp = Path(tempfile.gettempdir()) / f"fsvnet_{up.name}"
    tmp.write_bytes(up.getbuffer())
    video_path = str(tmp)
elif local.strip() and Path(local.strip()).exists():
    video_path = local.strip()

if video_path is None:
    st.info("Upload a clip or paste a path to begin.")
    st.stop()

with col_a:
    st.video(video_path)

run = st.button("Analyse", type="primary")
if not run:
    st.stop()

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

with st.spinner("Loading model…"):
    pipe, cfg = load_pipeline(ckpt, ref, device, frames, img_size, face_backend)
st.caption(f"Model geometry taken from the checkpoint: "
           f"**{cfg.data.img_size}px** crops, {len(pipe.model.branch_names)} "
           f"branches ({', '.join(pipe.model.branch_names)}).")

import torch  # noqa: E402

from fsvnet.explain import explain_clip, overlay_cam  # noqa: E402

with st.spinner("Decoding, cropping and scoring…"):
    try:
        result = pipe.run(video_path, explain=False, n_clips=n_clips,
                          make_figures=False)
    except Exception as e:
        st.error(f"Could not process this video: {e}")
        st.stop()

    x, mask, frames_u8, info = pipe._prepare(video_path, 0, 1)
    ex = explain_clip(pipe.model, x, mask, reference=pipe.reference,
                      do_cam=do_cam, do_occlusion=do_occ,
                      occlusion_mode=occ_mode)

p = result["fake_probability"]
band = result["confidence_band"]
verdict = result["verdict"]

# ---- verdict card --------------------------------------------------------
cls = ("verdict-unsure" if band in ("borderline", "weak")
       else ("verdict-fake" if verdict == "fake" else "verdict-real"))
label = ("LIKELY MANIPULATED" if verdict == "fake" else "LIKELY AUTHENTIC")
if band in ("borderline", "weak"):
    label = "INCONCLUSIVE"

st.markdown(f"""
<div class="verdict-card {cls}">
  <h2>{label}</h2>
  <p>p(fake) = <b>{p:.3f}</b> &nbsp;·&nbsp; confidence: <b>{band}</b>
     &nbsp;·&nbsp; {result['n_clips']} clip(s) averaged
     &nbsp;·&nbsp; face detected in {(info.get('face_ratio') or 0)*100:.0f}% of sampled frames</p>
</div>
""", unsafe_allow_html=True)

if band in ("borderline", "weak"):
    st.warning("The model is close to undecided. Treat this as *no evidence "
               "either way*, not as a weak yes.", icon="⚖️")

m1, m2, m3, m4 = st.columns(4)
m1.metric("p(fake)", f"{p:.3f}")
m2.metric("Clip spread", f"{np.std(result['clip_probabilities']):.3f}",
          help="Disagreement between clips of the same video. Large spread "
               "means the video is not uniformly real or fake.")
m3.metric("Sampling rate", f"{(info.get('sample_hz') or 0):.1f} Hz",
          help="Effective frame rate after striding — the temporal axis of the "
               "TSC analysis")
top_branch = (max(ex.attribution, key=lambda k: ex.attribution[k]["share"])
              if ex.attribution else "—")
m4.metric("Decisive branch", top_branch,
          help="By measured dependence, not by attention weight")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tabs = st.tabs(["Summary", "Where", "When", "Which branch",
                "Forensic measurements", "Reasoning"])

# ---- Summary -------------------------------------------------------------
with tabs[0]:
    for line in ex.summary_lines():
        st.markdown(f"- {line}")
    st.divider()
    st.markdown("**Per-clip scores**")
    st.bar_chart({"p(fake)": result["clip_probabilities"]}, height=180)
    st.caption("Each bar is one temporal window. The video-level score is their "
               "mean — the same aggregation used when the model is evaluated.")

# ---- Where ---------------------------------------------------------------
with tabs[1]:
    if ex.cam is None:
        st.info("Grad-CAM is off, or the spatial branch is disabled.")
    else:
        st.markdown("**Grad-CAM over the spatial (EfficientNet) branch.** "
                    "Brighter = pixels that pushed the *fake* logit up.")
        sel = ex.top_frames(6) if ex.occlusion is not None else list(range(6))
        sel = sorted(set(sel))[:6]
        cols = st.columns(len(sel))
        for c, t in zip(cols, sel):
            c.image(overlay_cam(ex.frames[t], ex.cam[t]),
                    caption=f"frame {t}")
        st.caption("This map reflects the semantic branch only. The frequency, "
                   "SRM and TSC branches have no meaningful spatial map — "
                   "drawing one for them would be decoration, not evidence.")

# ---- When ----------------------------------------------------------------
with tabs[2]:
    if ex.occlusion is None:
        st.info("Temporal occlusion is off.")
    else:
        d = ex.occlusion["delta"]
        st.markdown("**Per-frame occlusion.** Each frame is replaced in turn "
                    "and the fake-logit is re-read. Positive = that frame "
                    "carried evidence *for* manipulation.")
        st.bar_chart({"Δ fake-logit": d.tolist()}, height=240)
        tf = ex.top_frames(4)
        cols = st.columns(len(tf))
        for c, t in zip(cols, tf):
            c.image(ex.frames[t], caption=f"frame {t}  (Δ {d[t]:+.3f})",
                    use_container_width=True)
        if ex.frame_attention is not None:
            with st.expander("Temporal attention weights (the model's own gate)"):
                st.bar_chart({"attention": ex.frame_attention.tolist()}, height=180)
                st.caption("Attention is what the pooling *weighted*; occlusion "
                           "is what the answer *depended on*. They are not the "
                           "same quantity and can disagree.")

# ---- Which branch --------------------------------------------------------
with tabs[3]:
    if not ex.attribution:
        st.info("No attribution available.")
    else:
        import pandas as pd
        df = pd.DataFrame(ex.attribution).T
        df.index.name = "branch"
        show = df[["gate", "share", "delta_logit", "delta_prob"]].copy()
        show.columns = ["attention gate", "dependence share",
                        "Δ logit if removed", "Δ p(fake) if removed"]
        st.dataframe(show.style.format("{:.4f}"))
        st.bar_chart({"dependence share": df["share"].to_dict()}, height=220)
        dis = [k for k, v in ex.attribution.items() if v.get("disagrees")]
        if dis:
            st.warning(f"Gate and dependence disagree for: **{', '.join(dis)}**. "
                       "The gate says the fusion scaled that branch up; the "
                       "ablation says the answer barely moved without it. "
                       "Report the ablation.", icon="🔍")
        st.caption("Dependence is measured by zeroing each branch and re-reading "
                   "the logit. That is the number that belongs in a report; the "
                   "gate weight is a diagnostic beside it.")

# ---- Forensic measurements ----------------------------------------------
with tabs[4]:
    import pandas as pd

    from fsvnet.evidence import CONTEXT_ONLY, DESCRIPTOR_MEANING

    if not ex.descriptors:
        st.info("The TSC branch is disabled, so there are no measurements.")
    else:
        rows = []
        for k, v in ex.descriptors.items():
            pc = ex.percentiles.get(k)
            status = "—"
            if k in CONTEXT_ONLY:
                status = "context only"
            elif pc is None:
                status = "not calibrated"
            elif pc <= 5 or pc >= 95:
                status = "ANOMALOUS"
            else:
                status = "normal"
            rows.append({"measurement": k, "value": round(v, 4),
                         "percentile of real": ("—" if pc is None else f"{pc:.0f}"),
                         "status": status,
                         "what it means": DESCRIPTOR_MEANING.get(k, "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if not ex.percentiles or all(v is None for v in ex.percentiles.values()):
            st.warning("No reference distribution loaded, so these are raw "
                       "numbers with no interpretation. Build one with "
                       "`evidence.build_reference_stats()` over your training "
                       "reals.", icon="⚠️")

        if ex.tsc_map is not None:
            st.markdown("**Temporal Spectral Coherence map**")
            import matplotlib.pyplot as plt

            from fsvnet import viz
            viz.use_style()
            fig, ax = plt.subplots(figsize=(6, 3.4))
            im = ax.imshow(ex.tsc_map, aspect="auto", origin="lower",
                           cmap=viz.sequential_cmap())
            ax.set_xlabel("spatial frequency ring →")
            ax.set_ylabel("temporal frequency →")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
            st.caption("Real footage keeps energy flat up the right-hand side "
                       "(sensor noise is independent per frame). Generated "
                       "detail rolls off, because it is computed from a "
                       "temporally smooth latent.")

# ---- Reasoning -----------------------------------------------------------
with tabs[5]:
    if not do_reason:
        st.info("Written reasoning is off.")
    else:
        with st.spinner("Generating the written analysis…"):
            try:
                from fsvnet.evidence import build_evidence
                from fsvnet.vlm_reasoner import (EvidenceReasoner,
                                                 groundedness_check)
                pack = build_evidence(pipe.model, x, mask, video_path, cfg,
                                      pipe.reference, frames_u8,
                                      make_figures=False)
                expl = EvidenceReasoner(cfg).explain(pack, frames_u8)
                st.markdown(f"**Backend:** `{expl.backend}`")
                if expl.backend == "template":
                    st.info("No vision-language model is installed, so this is "
                            "the deterministic template — terse, but it cannot "
                            "hallucinate. `pip install transformers accelerate "
                            "bitsandbytes qwen-vl-utils` for the Qwen rung.",
                            icon="ℹ️")
                st.write(expl.text)
                with st.expander("Groundedness check"):
                    st.json(groundedness_check(expl.text, pack))
                    st.caption("A guard-rail, not an evaluation: it checks that "
                               "the text cites measured quantities and hedges "
                               "on borderline scores.")
                with st.expander("The exact prompt the model received"):
                    st.code(expl.prompt, language="text")
            except Exception as e:
                st.error(f"Reasoning layer failed: {e}")

st.divider()
st.caption(f"Checkpoint: `{Path(ckpt).name}` · device: {cfg.device} · "
           f"{cfg.data.num_frames} frames @ {cfg.data.img_size}px · "
           f"branches: {', '.join(pipe.model.branch_names)}")
