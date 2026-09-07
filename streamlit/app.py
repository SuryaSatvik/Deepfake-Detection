"""
AI Media Authenticity Analyzer
==============================

    streamlit run streamlit/app.py

Two deliberate design choices, both of which cost visual drama and buy trust:

1. **The verdict is not the biggest thing on the page.** A detector that scores
   0.85 AUC is wrong about one item in seven. A huge red FAKE and nothing else
   teaches a user to stop reading, which is the opposite of what a forensic
   tool should do. The verdict shares the top of the page with its confidence
   band, its calibration caveat, and the evidence.

2. **Model evidence and generated explanation are visually separate.**
   Evidence is measured (percentile-calibrated descriptors, region-energy
   series, TSC maps). Explanation is written text grounded in those
   measurements. Blending them is how a demo starts looking like it knows more
   than it does.

WHAT THIS APP WILL NOT DO
-------------------------
* **No audio detection.** Celeb-DF-v2 ships no audio, no audio model was
  trained, and offering an audio tab that returns something would be a lie.
  Audio uploads are accepted and explicitly refused with a reason.
* **No private chain-of-thought.** The Reasoning panel shows an evidence-
  grounded summary. If a VLM produced internal deliberation it is not
  surfaced.
* **No verdict without a checkpoint.** With no trained model present the app
  runs in EVIDENCE-ONLY mode: it still computes and shows the forensic
  measurements, and says plainly that there is no detector behind them.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import streamlit as st
except ImportError:                                        # pragma: no cover
    raise SystemExit("streamlit is not installed:  pip install streamlit")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="AI Media Authenticity Analyzer",
                   page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2.0rem; max-width: 1400px; }
  .vcard { border-radius: 10px; padding: 1.0rem 1.2rem; margin-bottom: .5rem;
           border: 1px solid rgba(128,128,128,.25); }
  .v-fake   { background: rgba(193,69,43,.10);  border-left: 5px solid #c1452b; }
  .v-real   { background: rgba(43,122,120,.10); border-left: 5px solid #2b7a78; }
  .v-unsure { background: rgba(245,159,0,.12);  border-left: 5px solid #f59f00; }
  .vcard h2 { margin: 0 0 .2rem 0; font-size: 1.45rem; }
  .vcard p  { margin: 0; opacity: .85; font-size: .9rem; }
  .caveat   { font-size: .84rem; opacity: .72; }
  .evbox    { background: rgba(128,128,128,.07); border-radius: 8px;
              padding: .7rem .9rem; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

MAX_UPLOAD_MB = 300
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


# ==========================================================================
# Robust loading — every import that can fail, fails into a message
# ==========================================================================

@st.cache_resource(show_spinner=False)
def load_core():
    """Import what is available. Never raises; reports what is missing."""
    state = {"torch": None, "cuda": False, "fsvnet": False, "cv2": False,
             "errors": []}
    try:
        import torch
        state["torch"] = torch.__version__
        state["cuda"] = torch.cuda.is_available()
    except Exception as e:
        state["errors"].append(f"torch unavailable ({e}) — CPU/evidence-only mode")
    try:
        import cv2  # noqa: F401
        state["cv2"] = True
    except Exception as e:
        state["errors"].append(f"opencv unavailable ({e}) — LMF cannot run")
    try:
        from fsvnet import lmf, celebdf_frames  # noqa: F401
        state["fsvnet"] = True
    except Exception as e:
        state["errors"].append(f"fsvnet import failed ({e})")
    return state


@st.cache_resource(show_spinner=False)
def load_template():
    p = ROOT / "data" / "manifests" / "lmf_template.npy"
    return np.load(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_reference():
    """Training-real descriptor distributions, for percentile calibration."""
    import pandas as pd
    out = {}
    for name, f in [("lmf", "lmf_descriptors_08f.csv"), ("tsc", "tsc_descriptors_08f.csv")]:
        p = ROOT / "results" / f
        if p.exists():
            d = pd.read_csv(p, dtype={"clip_name": str})
            out[name] = d[(d.split == "train") & (d.label == 0)]
    return out


CONDITION_LABEL = {
    "B0": "FSV-Net conventional (spatial + frequency + SRM)",
    "C1": "FSV-Net/DFE + TSC",
    "C2": "FSV-Net/DFE + LMF",
    "C3": "FSV-Net/DFE full (TSC + LMF)  — proposed",
}


@st.cache_resource(show_spinner=False)
def find_checkpoints():
    # torch may be absent (evidence-only install). That must degrade to "no
    # models available", not crash the page before it renders.
    """
    Discover trained conditions and what each one NEEDS.

    A checkpoint with use_lmf=True cannot be scored on an input with no facial
    landmarks -- its LMF branch would receive zeros and the verdict would come
    from a crippled model. The UI uses `needs_lmf` to disable those rather than
    quietly returning a degraded number.
    """
    try:
        import torch
    except Exception:
        return []
    out = []
    d = ROOT / "models" / "checkpoints"
    if not d.exists():
        return out
    for f in sorted(d.glob("fsvnet_dfe_*.pt")):
        cid = f.stem.replace("fsvnet_dfe_", "")
        try:
            blob = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        out.append({
            "id": cid, "path": f,
            "label": CONDITION_LABEL.get(cid, cid),
            "use_tsc": bool(blob.get("use_tsc", False)),
            "needs_lmf": bool(blob.get("use_lmf", False)),
            "epoch": blob.get("epoch"),
            "metric": blob.get("metric"),
            "monitor": blob.get("monitor", "val_auc"),
            "cfg": blob.get("config") or {},
        })
    return out


CORE = load_core()
TEMPLATE = load_template()
REFERENCE = load_reference()
CKPTS = find_checkpoints()


# ==========================================================================
# Sidebar
# ==========================================================================

st.sidebar.title("🔍 Authenticity Analyzer")
st.sidebar.caption("Forensic evidence first, verdict second.")

n_frames = st.sidebar.slider("Frames to analyse", 4, 32, 16, step=4)
unsure_lo, unsure_hi = st.sidebar.slider(
    "UNCERTAIN band", 0.0, 1.0, (0.40, 0.60), step=0.05,
    help="Scores inside this band are reported as UNCERTAIN rather than forced "
         "to a side. Widen it if the model's calibration is poor.")

st.sidebar.divider()
st.sidebar.markdown("**Environment**")
st.sidebar.text(f"torch   {CORE['torch'] or 'missing'}")
st.sidebar.text(f"device  {'cuda' if CORE['cuda'] else 'cpu'}")
st.sidebar.text(f"opencv  {'ok' if CORE['cv2'] else 'missing'}")
st.sidebar.text(f"ckpts   {len(CKPTS)}")
for e in CORE["errors"]:
    st.sidebar.warning(e, icon="⚠️")

EVIDENCE_ONLY = not CKPTS
if EVIDENCE_ONLY:
    st.sidebar.error(
        "No trained checkpoint found — running in EVIDENCE-ONLY mode. "
        "Forensic measurements are shown; no verdict is produced, because "
        "there is no detector to produce one.", icon="🚫")


# ==========================================================================
# Header
# ==========================================================================

st.title("AI Media Authenticity Analyzer")
st.caption("Upload an image or a video. You get the measurements, then the verdict.")

with st.expander("What this tool can and cannot tell you", expanded=EVIDENCE_ONLY):
    st.markdown("""
**Can**: measure frame-to-frame forensic quantities and say how unusual they
are relative to the real videos the model was trained on; show *where* in the
face and *when* in the clip the anomaly sits; give a calibrated probability
when a trained checkpoint is loaded.

**Cannot**:

* **Analyse audio.** No audio model was trained — the training data has none.
  Audio files are refused rather than answered.
* **Prove** a media item is authentic. A low score means *no evidence of the
  artefacts this model measures*, which is not the same as genuine.
* **Explain the language model.** Grad-CAM covers the CNN spatial branch only.
* **Transfer freely.** The detector is trained on Celeb-DF-v2 face swaps.
  On a fully AI-generated video, or an unfamiliar codec, treat it as untested.

Every number shown is measured at run time. Nothing on this page is a
placeholder.
    """)


# ==========================================================================
# Decoding
# ==========================================================================

def read_frames(path: Path, is_video: bool, want: int):
    """
    Uniform temporal sampling that preserves order.
    Returns (frames (T,H,W,3) uint8, timestamps in seconds).
    """
    if not is_video:
        from PIL import Image
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return a[None], [0.0]

    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("the file could not be decoded — it may be corrupt "
                           "or use a codec OpenCV was not built with")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    frames, ts = [], []
    if total > 0:
        for i in np.unique(np.linspace(0, total - 1, min(want, total)).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); ts.append(i / fps)
    else:                                   # streams with no frame count
        k = 0
        while len(frames) < want:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); ts.append(k / fps); k += 1
    cap.release()
    if not frames:
        raise RuntimeError("no frames could be decoded from this file")
    return np.stack(frames), ts


# ==========================================================================
# Turning an input into frames the model can score
# ==========================================================================

@st.cache_data(show_spinner=False)
def crop_faces(frames: np.ndarray, size: int = 256):
    """
    Detect and crop faces from an arbitrary clip.

    Returns (crops, face_ratio, backend, note). `face_ratio` is the fraction of
    probed frames where a face was actually found. It is surfaced because a low
    ratio means the model is scoring background, and a verdict on background is
    meaningless however confident it looks.
    """
    try:
        from fsvnet.faces import build_detector, crop_clip
        det = build_detector("auto")
        res = crop_clip(frames, det, img_size=size, margin=0.30, detect_every=8)
        note = ""
        if res.face_ratio < 0.6:
            note = (f"A face was found in only {res.face_ratio*100:.0f}% of probed "
                    "frames (backend: "
                    f"{res.backend}). The crop may be mostly background, so treat "
                    "any verdict below as unreliable rather than confident. "
                    "Installing `mediapipe` gives a much better detector.")
        return res.frames, float(res.face_ratio), res.backend, note
    except Exception as e:
        import cv2
        out = np.stack([cv2.resize(f, (size, size)) for f in frames])
        return out, 0.0, "none", (
            f"Face detection unavailable ({type(e).__name__}: {e}); the whole "
            "frame was resized instead. The model was trained on face crops, so "
            "this run is approximate.")


@st.cache_data(show_spinner=False)
def load_dataset_clip(video_path: str, landmark_path: str, n_frames: int):
    """A Celeb-DF-v2 clip: already cropped, and it HAS the 81-point landmarks."""
    from fsvnet import celebdf_frames as cfx
    return cfx.load_clip(video_path, landmark_path, n_frames=n_frames, img_size=256)


# ==========================================================================
# Input: an uploaded file, or a clip from the evaluation set
# ==========================================================================
# Two sources, because they differ in ONE way that decides which models can
# honestly be run: a Celeb-DF-v2 clip ships 81-point facial landmarks, an
# arbitrary upload does not. The LMF branch is built on those landmarks, so a
# model trained with use_lmf=True cannot be scored on an upload without
# feeding its forensic branch zeros. Rather than hide that, the UI restricts
# the model list and says why.

@st.cache_data(show_spinner=False)
def load_test_clips():
    try:
        from fsvnet import celebdf_frames as cfx
        import pandas as pd
        man = ROOT / "data" / "manifests" / "celebdf_v2_identity.csv"
        if not man.exists():
            return None
        df = cfx.load_manifest(man, ROOT)
        te = df[df.split == "test"].copy()
        return te[["clip_name", "video_path", "landmark_path", "label_name", "subset"]]
    except Exception:
        return None


TEST_CLIPS = load_test_clips()
SOURCE = st.radio(
    "Input",
    ["Upload an image or video",
     "Pick a Celeb-DF-v2 test clip (has landmarks — enables every model)"],
    index=0, horizontal=False,
    disabled=(TEST_CLIPS is None) and False)

crops = landmarks = None
timestamps: list = []
face_ratio, source_desc, truth = None, "", None

# ---------------------------------------------------------------- upload ---
if SOURCE.startswith("Upload"):
    up = st.file_uploader(
        "Upload image / video   (audio is accepted but explicitly not supported)",
        type=sorted(x[1:] for x in (VIDEO_EXT | IMAGE_EXT | AUDIO_EXT)))

    if up is None:
        st.info("Waiting for a file. Nothing is analysed until you upload one.")
        st.stop()

    suffix = Path(up.name).suffix.lower()
    size_mb = up.size / 1e6

    if size_mb > MAX_UPLOAD_MB:
        st.error(f"`{up.name}` is {size_mb:.0f} MB; the limit is {MAX_UPLOAD_MB} MB. "
                 "Trim the clip and try again — a few seconds is enough.", icon="🚫")
        st.stop()

    if suffix in AUDIO_EXT:
        st.error(
            f"**Audio is not supported, and this tool will not guess.**\n\n"
            f"`{up.name}` is an audio file. No audio model was trained in this "
            "project — Celeb-DF-v2 contains no audio track at all. Returning a "
            "verdict here would be fabricating one.\n\n"
            "For audio deepfake detection you need a model trained on something "
            "like ASVspoof or WaveFake; this is not that system.", icon="🚫")
        st.stop()

    if suffix not in VIDEO_EXT | IMAGE_EXT:
        st.error(f"Unsupported format `{suffix}`. Supported: "
                 f"{', '.join(sorted(VIDEO_EXT | IMAGE_EXT))}", icon="🚫")
        st.stop()

    tmp = Path(tempfile.gettempdir()) / f"authenticity_{int(time.time())}{suffix}"
    tmp.write_bytes(up.getbuffer())
    is_video = suffix in VIDEO_EXT
    try:
        with st.spinner("Decoding…"):
            frames, timestamps = read_frames(tmp, is_video, n_frames)
    except Exception as e:
        st.error(f"**Could not read `{up.name}`.**\n\n`{type(e).__name__}: {e}`\n\n"
                 "The file may be corrupt, truncated, or use an unsupported codec. "
                 "Nothing else on this page ran.", icon="🚫")
        st.stop()
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    with st.spinner("Detecting and cropping faces…"):
        crops, face_ratio, backend, note = crop_faces(frames, 256)
    landmarks = None
    source_desc = (f"`{up.name}` · {len(crops)} frames · {size_mb:.1f} MB · "
                   f"face detector: {backend} ({face_ratio*100:.0f}% of probed frames)")
    if note:
        st.warning(note, icon="⚠️")
    st.info("An uploaded file carries no 81-point landmarks, so the **LMF branch "
            "cannot run** and models trained with it are disabled below. "
            "This is a missing input, not a negative finding.", icon="ℹ️")

# --------------------------------------------------------- dataset clip ---
else:
    if TEST_CLIPS is None or not len(TEST_CLIPS):
        st.error("No manifest found. Run `python scripts/build_celebdf_manifest.py`.",
                 icon="🚫")
        st.stop()
    names = TEST_CLIPS.clip_name.tolist()
    default = names.index("id36_0000") if "id36_0000" in names else 0
    pick = st.selectbox(
        f"Test-split clip ({len(names)} available — held out, identity-disjoint)",
        names, index=default)
    row = TEST_CLIPS[TEST_CLIPS.clip_name == pick].iloc[0]
    truth = row.label_name
    with st.spinner("Loading clip…"):
        crops, landmarks, meta = load_dataset_clip(row.video_path, row.landmark_path,
                                                   n_frames)
        timestamps = meta["timestamp_s_approx"]
    face_ratio = 1.0
    source_desc = (f"`{pick}` · {row.subset} · {len(crops)} frames · "
                   f"landmarks: {'yes' if landmarks is not None else 'no'}")
    st.caption("This clip is from the held-out test split — no model here was "
               "trained on it, or on any clip sharing its identity.")

st.success(f"Ready — {source_desc}")
if truth:
    st.caption(f"Ground truth for this clip: **{truth.upper()}** "
               "(shown because it comes from the labelled test split; "
               "an upload has no ground truth to show)")


# ==========================================================================
# Evidence
# ==========================================================================

def percentile_of(name: str, value: float, table: str):
    ref = REFERENCE.get(table)
    if ref is None or name not in ref.columns or not np.isfinite(value):
        return None
    base = ref[name].dropna().values
    return float((base < value).mean() * 100) if len(base) else None


@st.cache_data(show_spinner=False)
def compute_evidence(crops: np.ndarray, landmarks, _tpl):
    ev = {"lmf": None, "tsc": None, "errors": []}
    try:
        # tsc_np is deliberately torch-free, so the evidence panel still works
        # on a machine where the detector cannot load at all.
        from fsvnet.tsc_np import tsc_descriptors_np, tsc_map_np
        m, sp = tsc_map_np(crops, n_bins=64)
        ev["tsc"] = {"descriptors": tsc_descriptors_np(m, sp), "map": m}
    except Exception as e:
        ev["errors"].append(f"TSC unavailable: {type(e).__name__}: {e}")
    if landmarks is not None and _tpl is not None:
        try:
            from fsvnet import lmf as LMF
            r = LMF.analyse_clip(crops, landmarks, _tpl, size=256, keep_maps=True)
            ev["lmf"] = {"descriptors": r.descriptors, "energy": r.energy,
                         "residual": r.residual, "masks": r.masks}
        except Exception as e:
            ev["errors"].append(f"LMF unavailable: {type(e).__name__}: {e}")
    elif landmarks is None:
        ev["errors"].append(
            "LMF needs 81-point facial landmarks and none were produced for "
            "this upload, so the geometry channel is unavailable. That is a "
            "missing input, not a negative finding.")
    return ev


with st.spinner("Measuring forensic evidence…"):
    EV = compute_evidence(crops, landmarks, TEMPLATE)

for e in EV["errors"]:
    st.info(e, icon="ℹ️")


# ==========================================================================
# Verdict
# ==========================================================================

# ==========================================================================
# Which models can honestly be run on THIS input
# ==========================================================================

HAVE_LM = landmarks is not None
usable = [c for c in CKPTS if HAVE_LM or not c["needs_lmf"]]
blocked = [c for c in CKPTS if c["needs_lmf"] and not HAVE_LM]

st.divider()
st.subheader("Analyse")

if not CKPTS:
    st.error("No trained checkpoint in `models/checkpoints/`. Running in "
             "evidence-only mode: the measurements below are still computed, "
             "but no verdict is produced because there is no detector to "
             "produce one.", icon="🚫")
    picked = []
else:
    labels = {f"{c['id']} — {c['label']}": c for c in usable}
    default = [k for k in labels if k.startswith(("B0", "C3"))] or list(labels)[:1]
    chosen = st.multiselect(
        "Models to run (each was trained on the identity-disjoint train split)",
        list(labels), default=default)
    picked = [labels[k] for k in chosen]
    if blocked:
        st.caption("Disabled for this input — needs 81-point landmarks: "
                   + ", ".join(f"`{c['id']}`" for c in blocked)
                   + ". Pick a Celeb-DF-v2 clip above to enable them.")
    cols = st.columns(max(len(usable), 1))
    for col, c in zip(cols, usable):
        col.caption(f"**{c['id']}** · best {c['monitor']} "
                    f"{c['metric']:.4f} @ epoch {c['epoch']}"
                    if c.get("metric") is not None else f"**{c['id']}**")

go = st.button("Run analysis", type="primary", disabled=not picked)


@st.cache_resource(show_spinner=False)
def _load_model(path_str: str, use_tsc: bool, use_lmf: bool, n_frames: int, img: int):
    import torch
    from fsvnet import Config
    from fsvnet.model import build_model
    cfg = Config()
    cfg.data.num_frames, cfg.data.img_size = n_frames, img
    cfg.model.use_tsc, cfg.model.use_lmf = use_tsc, use_lmf
    cfg.model.pretrained = False          # weights come from the checkpoint
    m = build_model(cfg)
    blob = torch.load(path_str, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if "state_dict" in blob else blob
    missing, unexpected = m.load_state_dict(sd, strict=False)
    dev = "cuda" if CORE["cuda"] else "cpu"
    return m.to(dev).eval(), list(missing), list(unexpected)


def run_models(picked):
    """Score the input with each selected checkpoint. Never fabricates."""
    out = {}
    if not picked:
        return out
    try:
        import torch, cv2
        from fsvnet.dataset_frames import normalise
        from fsvnet.lmf import DESCRIPTOR_NAMES
    except Exception as e:
        st.error(f"Inference path unavailable ({type(e).__name__}: {e}).", icon="🚫")
        return out

    dev = "cuda" if CORE["cuda"] else "cpu"
    small = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
                      for f in crops])
    clip = normalise(small)[None].to(dev)

    pack = None
    if EV.get("lmf") is not None:
        pack = {"energy": torch.tensor(EV["lmf"]["energy"])[None].to(dev),
                "descriptors": torch.tensor(
                    [EV["lmf"]["descriptors"][k] for k in DESCRIPTOR_NAMES],
                    dtype=torch.float32)[None].to(dev)}

    for c in picked:
        try:
            m, missing, unexpected = _load_model(str(c["path"]), c["use_tsc"],
                                                 c["needs_lmf"], len(crops), 224)
            if missing:
                st.warning(f"{c['id']}: {len(missing)} weights missing from the "
                           "checkpoint — the architecture and the file disagree. "
                           "Treat this result with suspicion.", icon="⚠️")
            t0 = time.time()
            with torch.no_grad():
                prob = float(m.predict_proba(clip, lmf=pack if c["needs_lmf"] else None)[0, 1])
            out[c["id"]] = {"p_fake": prob, "ms": (time.time() - t0) * 1000,
                            "label": c["label"], "needs_lmf": c["needs_lmf"],
                            "use_tsc": c["use_tsc"],
                            "val": c.get("metric"), "epoch": c.get("epoch")}
        except Exception as e:
            st.warning(f"{c['id']} could not be run: {type(e).__name__}: {e}",
                       icon="⚠️")
    return out


PREDS = run_models(picked) if go else {}


def verdict_of(p):
    if p is None:
        return "NO VERDICT", "v-unsure"
    if p < unsure_lo:
        return "REAL", "v-real"
    if p > unsure_hi:
        return "FAKE", "v-fake"
    return "UNCERTAIN", "v-unsure"


if go and not PREDS:
    st.markdown('<div class="vcard v-unsure"><h2>NO VERDICT</h2><p>No selected '
                'model produced a score. The measurements below still stand.</p>'
                '</div>', unsafe_allow_html=True)
elif PREDS:
    cols = st.columns(len(PREDS))
    for col, (cid, r) in zip(cols, PREDS.items()):
        v, cls = verdict_of(r["p_fake"])
        with col:
            st.markdown(
                f'<div class="vcard {cls}"><h2>{v} &nbsp;·&nbsp; {r["p_fake"]*100:.0f}%</h2>'
                f'<p>{cid} — {r["label"]}<br>{r["ms"]:.0f} ms · '
                f'val AUC {r["val"]:.4f}</p></div>', unsafe_allow_html=True)
            st.progress(min(max(r["p_fake"], 0.0), 1.0))
    if truth:
        agree = sum(1 for r in PREDS.values()
                    if verdict_of(r["p_fake"])[0] == truth.upper())
        st.caption(f"Ground truth **{truth.upper()}** — "
                   f"{agree} of {len(PREDS)} model(s) agree.")
    st.markdown(
        '<p class="caveat">The percentage is the model&rsquo;s P(manipulated) on '
        'this input, not a probability that the media is a deepfake in the real '
        'world. These models were trained on Celeb-DF-v2 face swaps; on any '
        'other manipulation type, or an unfamiliar codec, treat them as '
        'untested.</p>', unsafe_allow_html=True)

    if len(PREDS) > 1:
        import pandas as pd
        st.markdown("#### Model comparison")
        rows = []
        for cid, r in PREDS.items():
            v, _ = verdict_of(r["p_fake"])
            rows.append({"Model": cid, "Verdict": v,
                         "P(fake)": f"{r['p_fake']*100:.1f}%",
                         "Forensic channels": ", ".join(
                             [x for x in ["TSC" if r["use_tsc"] else None,
                                          "LMF" if r["needs_lmf"] else None] if x]
                         ) or "none (spatial + frequency + SRM only)",
                         "Val AUC": f"{r['val']:.4f}",
                         "Inference": f"{r['ms']:.0f} ms"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("All models see the identical frames. Disagreement between "
                   "them is information — it usually means the forensic "
                   "channels and the spatial branch are reading the clip "
                   "differently, which the evidence tabs below let you inspect.")
else:
    st.info("Choose your models and press **Run analysis**.")

# ==========================================================================
# Evidence panels
# ==========================================================================

st.divider()
tabs = st.tabs(["Sampled frames", "Artifact evidence", "Where (spatial)",
                "When (temporal)", "Explanation"])

with tabs[0]:
    st.caption("Uniform temporal sampling, order preserved. "
               "Timestamps come from the container where available.")
    ncol = min(8, len(crops))
    for start in range(0, len(crops), ncol):
        cs = st.columns(ncol)
        for c, j in zip(cs, range(start, min(start + ncol, len(crops)))):
            c.image(crops[j], caption=f"t={timestamps[j]:.2f}s" if j < len(timestamps) else "",
                    use_container_width=True)

with tabs[1]:
    st.markdown("#### Measured quantities, and how unusual each one is")
    st.caption("Percentile is the position among the **training real** videos. "
               "Below the 5th or above the 95th is flagged. These are "
               "measurements, not the model's decision.")
    import pandas as pd
    rows = []
    for tag, table in [("LMF", "lmf"), ("TSC", "tsc")]:
        blk = EV["lmf"] if tag == "LMF" else EV["tsc"]
        if not blk:
            continue
        for k, v in blk["descriptors"].items():
            pct = percentile_of(k, float(v), table)
            rows.append({"channel": tag, "descriptor": k, "value": round(float(v), 4),
                         "percentile vs real": "n/a" if pct is None else f"{pct:.0f}",
                         "flag": "ANOMALOUS" if (pct is not None and (pct < 5 or pct > 95)) else ""})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No forensic channel could be computed for this file.")
    if not REFERENCE:
        st.warning("No reference distribution found, so percentiles are unavailable. "
                   "Run `scripts/extract_lmf.py` and `scripts/extract_tsc.py` to build "
                   "one. Raw values alone are hard to interpret.", icon="⚠️")

with tabs[2]:
    if EV["lmf"] is not None:
        st.markdown("#### Where the motion-compensated residual energy sits")
        st.caption("Frames are warped into a canonical face frame first, so head "
                   "motion is removed and the regions mean the same thing in "
                   "every frame.")
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = np.abs(EV["lmf"]["residual"]).mean(axis=0)
        m = EV["lmf"]["masks"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(crops[0]); ax[0].set_title("first frame")
        ov = np.zeros((256, 256, 3)); ov[..., 1] = m["inner"]; ov[..., 0] = m["boundary"]
        ax[1].imshow(crops[0]); ax[1].imshow(ov, alpha=.3)
        ax[1].set_title("inner face / blend seam")
        ax[2].imshow(res, cmap="magma", vmax=max(np.percentile(res, 99.3), 1e-6))
        ax[2].set_title("mean |residual|")
        for a in ax: a.axis("off")
        st.pyplot(fig, use_container_width=True)
        st.caption("This is a **measurement**, not a saliency map. It shows where "
                   "pixel change concentrated after motion was removed — it does "
                   "not show what the classifier attended to.")
    else:
        st.info("Spatial evidence needs the 81-point landmarks. Pick a "
                "Celeb-DF-v2 clip above to see this panel.")

with tabs[3]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drew = False
    if EV["lmf"] is not None:
        e = np.log10(np.maximum(EV["lmf"]["energy"], 1e-12))
        ts = timestamps[1:len(e) + 1] if len(timestamps) > len(e) else np.arange(len(e))
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for j, nm in enumerate(["inner face", "blend seam", "outside"]):
            ax.plot(ts[:len(e)], e[:, j], lw=1.8, label=nm)
        ax.set_xlabel("time (s)"); ax.set_ylabel("log10 residual energy")
        ax.legend(frameon=False); ax.grid(alpha=.25)
        ax.set_title("Region energy over time")
        st.pyplot(fig, use_container_width=True); drew = True
    if EV["tsc"] is not None:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        im = ax.imshow(EV["tsc"]["map"], aspect="auto", origin="lower", cmap="magma")
        ax.set_xlabel("radial frequency ring"); ax.set_ylabel("temporal frequency")
        ax.set_title("Temporal spectral coherence map"); fig.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True); drew = True
    if not drew:
        st.info("No temporal channel was available for this file.")

MEANING = {
    "inner_outer_contrast": ("frame-to-frame change inside the face mask "
                             "relative to outside it",
                             "a swapped face is decoded from a temporally smooth "
                             "latent, so it changes LESS between frames than real "
                             "sensor detail does"),
    "boundary_excess":      ("extra change concentrated on the blend seam",
                             "the boundary where a swapped face is composited "
                             "flickers relative to its surroundings"),
    "inner_instability":    ("how bursty the change inside the face is over time",
                             "generators are unstable in bursts; a camera sensor "
                             "is steady"),
    "contrast_drift":       ("drift in that inside/outside ratio across the clip",
                             "a blend that is re-estimated over time drifts"),
    "seam_periodicity":     ("a repeating rate in the seam's flicker",
                             "some blending pipelines re-estimate per keyframe, "
                             "leaving a rate"),
    "geometric_residual":   ("how far the face departs from rigid motion",
                             "high values mean expression, a landmark failure, or "
                             "warped geometry"),
    "geometric_jitter":     ("frame-to-frame instability of that geometry", ""),
    "temporal_whiteness":   ("whether fine detail flickers like sensor noise",
                             "real camera grain is temporally white; generated "
                             "detail is over-smooth"),
    "temporal_peakiness":   ("energy concentrated at one flicker rate",
                             "blending and colour correction leave a rate here"),
    "flicker_index":        ("too much or too little fine-scale flicker", ""),
    "fine_ring_acf1":       ("frame-to-frame correlation of fine detail",
                             "near 1 means over-smooth, i.e. computed rather than "
                             "captured"),
}


with tabs[4]:
    st.markdown("#### Why — the verdict, traced to measurements")

    anomalies = []
    for tag, table in [("LMF", "lmf"), ("TSC", "tsc")]:
        blk = EV["lmf"] if tag == "LMF" else EV["tsc"]
        if not blk:
            continue
        for k, v in blk["descriptors"].items():
            pct = percentile_of(k, float(v), table)
            if pct is not None and (pct < 5 or pct > 95):
                anomalies.append((tag, k, float(v), pct))
    anomalies.sort(key=lambda a: -abs(a[3] - 50))

    if PREDS:
        for cid, r in PREDS.items():
            v, _ = verdict_of(r["p_fake"])
            ch = [x for x in ["TSC" if r["use_tsc"] else None,
                              "LMF" if r["needs_lmf"] else None] if x]
            branches = ("the spatial, frequency and SRM branches" if not ch
                        else "those branches plus " + " and ".join(ch))
            st.markdown(f"**{cid}** called this **{v}** at "
                        f"{r['p_fake']*100:.0f}%, using {branches}.")
        if truth:
            st.markdown(f"Ground truth for this clip is **{truth.upper()}**.")
    else:
        st.markdown("**No detector was run**, so there is no verdict to explain. "
                    "The measurements below stand on their own.")

    st.markdown("---")
    if anomalies:
        st.markdown(f"**{len(anomalies)} measured quantit"
                    f"{'y is' if len(anomalies)==1 else 'ies are'} outside the "
                    "5th–95th percentile of the real videos this model was "
                    "trained on:**")
        for tag, k, val, pct in anomalies:
            what, why = MEANING.get(k, ("", ""))
            hi = "unusually high" if pct > 95 else "unusually low"
            st.markdown(
                f"- **`{k}`** = {val:.4f} — **{pct:.0f}th percentile**, {hi}.  \n"
                f"  *{what}.*" + (f" {why.capitalize()}." if why else ""))
    else:
        st.markdown(
            "**No measured quantity fell outside the 5th–95th percentile of real "
            "training videos.**  \nThat is *not* proof of authenticity — it means "
            "this particular set of measurements found nothing unusual. A "
            "manipulation this model has never seen would look exactly like this.")

    st.markdown("---")
    st.markdown("**What this explanation is, and is not**")
    st.markdown(
        "- Every line above is a **measurement** taken from this input and "
        "placed against the training reals. Nothing is generated prose.\n"
        "- The verdict comes from the network; the measurements come from the "
        "forensic operators. They can disagree, and when they do that is worth "
        "more than either alone.\n"
        "- No private model reasoning is surfaced, and none is invented.")
    st.markdown(
        '<p class="caveat">A low score means no evidence of the artefacts this '
        'model measures. It is not a certificate of authenticity.</p>',
        unsafe_allow_html=True)

st.divider()
st.caption(
    f"Frames analysed: {len(crops)} · device: {'cuda' if CORE['cuda'] else 'cpu'} · "
    f"uncertain band: {unsure_lo:.2f}–{unsure_hi:.2f} · "
    f"mode: {'evidence-only' if EVIDENCE_ONLY else 'detector + evidence'}")
