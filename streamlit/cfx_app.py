"""
cfx_app.py
==========

The CFX front end.  `streamlit run streamlit/cfx_app.py`

A separate file from `streamlit/app.py` on purpose -- that app works and is what
the earlier write-up describes, so this one is additive rather than a rewrite.

What the viewer sees
--------------------
    verdict (or "I can't tell")
    the confidence number and its band
    ONE explanation, in the register they chose
    the technical <-> layman toggle
    the figure for that one explanation

and, collapsed under "Audit", everything else: which model answered and why,
the solidity breakdown, exact Shapley values and interactions, every candidate
explanation with its faithfulness and its margin over random, the conformal
band, the assumed prior, and the raw card JSON to download.

Three things this UI deliberately does not do
---------------------------------------------
1. **The toggle does not call the model.** Both registers were generated when the
   clip was analysed and cached on the card. Switching is a dictionary lookup.
   That keeps it instant and keeps Qwen from being loaded twice on an 8 GB card.
2. **Model selection is not a setting.** Arbitration picks which specialist
   answers; exposing that as a dropdown would hand the contribution to a
   checkbox. The audit panel shows what it chose and why.
3. **The verdict is not the biggest thing on the page.** At this detector's
   honest accuracy it does not deserve to be. The confidence and its band are
   given the same weight, and the abstain message is given more.

Crop size is also not a setting: the frequency and TSC branches precompute a
radial ring index for one input size, and feeding them another raises. The
pipeline adopts the checkpoint's `img_size`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="FSV-Net / CFX", layout="wide")

FIT = ROOT / "results" / "cfx"
CARDS = FIT / "cards"


# ==========================================================================
# loading
# ==========================================================================


@st.cache_resource(show_spinner="loading the specialist panel ...")
def load_pipeline(cpu: bool):
    import torch
    from fsvnet import Config
    from fsvnet.arbitrate import ConfidenceStack, ReliabilityMatrix
    from fsvnet.cfx import CFXConfig, CFXPipeline
    from fsvnet.evidence import ReferenceStats
    from fsvnet.model import build_model
    from fsvnet.selective import AbstentionPolicy

    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    flags = {"B0": (False, False), "C1": (True, False),
             "C2": (False, True), "C3": (True, True)}
    models = {}
    for cid, (tsc, lmf) in flags.items():
        ck = ROOT / "models" / "checkpoints" / f"fsvnet_dfe_{cid}.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        c = Config()
        c.device = device
        c.data.num_frames = 16
        # the checkpoint's own geometry wins -- see the module docstring
        c.data.img_size = int((blob.get("config") or {}).get("data", {})
                              .get("img_size", 224))
        c.model.use_tsc = blob.get("use_tsc", tsc)
        c.model.use_lmf = blob.get("use_lmf", lmf)
        m = build_model(c).to(device)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        models[cid] = m

    if not models:
        return None, device, "no checkpoints found in models/checkpoints/"

    matrix = (ReliabilityMatrix.load(FIT / "evidence_reliability.json")
              if (FIT / "evidence_reliability.json").exists() else ReliabilityMatrix())
    policy = (AbstentionPolicy.load(FIT / "abstention_policy.json")
              if (FIT / "abstention_policy.json").exists() else None)
    stack = (ConfidenceStack.load(FIT / "confidence_stack.json")
             if (FIT / "confidence_stack.json").exists() else None)
    ref_p = ROOT / "data" / "manifests" / "reference_stats.json"
    reference = ReferenceStats.load(ref_p) if ref_p.exists() else None

    cfx = CFXConfig()
    if device == "cpu":
        cfx = cfx.cpu_fallback()
    reasoner = None
    try:
        from fsvnet.vlm_reasoner import EvidenceReasoner
        cc = Config()
        cc.device = device
        reasoner = EvidenceReasoner(cc)
    except Exception:
        pass

    pipe = CFXPipeline(models=models, matrix=matrix, stack=stack, policy=policy,
                       cfx=cfx, reasoner=reasoner, reference=reference)
    missing = [n for n, ok in (("abstention policy", policy is not None),
                               ("confidence stack", stack is not None),
                               ("reference distribution", reference is not None))
               if not ok]
    warn = ("run scripts/cfx_fit.py -- missing: " + ", ".join(missing)) if missing else ""
    return pipe, device, warn


def load_saved_cards():
    if not CARDS.exists():
        return {}
    out = {}
    for f in sorted(CARDS.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


# ==========================================================================
# rendering
# ==========================================================================

DECISION_COLOUR = {"fake": "#b3261e", "real": "#1b7f4f", "abstain": "#6b6b6b"}


def render_user_surface(user: dict, register: str) -> None:
    dec = str(user["decision"])
    p = float(user["confidence_number"])
    lo, hi = user["confidence_band"]

    left, right = st.columns([3, 2])
    with left:
        label = {"fake": "Likely manipulated", "real": "Likely genuine",
                 "abstain": "Not sure enough to say"}[dec]
        st.markdown(f"#### <span style='color:{DECISION_COLOUR[dec]}'>{label}</span>",
                    unsafe_allow_html=True)
        st.write(user["text"])
    with right:
        st.metric("confidence score", f"{p:.3f}")
        st.caption(f"answers are only given outside the band "
                   f"[{lo:.3f}, {hi:.3f}]")
        # a bar, not a gauge: a gauge implies precision this number does not have
        st.progress(min(max(p, 0.0), 1.0))

    if user.get("explanation_available"):
        st.caption(f"Explanation shown: {user['explanation_of']}")
    elif user.get("explanation_of"):
        st.info(user["explanation_of"])
    if user.get("figure") and Path(str(user["figure"])).exists():
        st.image(str(user["figure"]))


def render_audit(audit: dict) -> None:
    v = audit["verdict"]
    c1, c2, c3 = st.columns(3)
    c1.metric("answering model", v["answering_model"])
    c2.metric("raw p(fake)", f"{v['p_fake_raw']:.3f}")
    c3.metric("calibrated p(fake)", f"{v['p_fake_calibrated']:.3f}")
    st.caption(f"alpha = {v['alpha']}, assumed fake prior = {v['assumed_prior']}, "
               f"uncertainty = {v['uncertainty']:.3f}, "
               f"cross-model agreement = {v['agreement']:.3f}")
    st.write("**Why this model answered**")
    st.write(v.get("routing_reason", ""))

    per_ch = (audit.get("routing") or {}).get("per_channel") or []
    if per_ch:
        import pandas as pd
        st.write("**Solidity, factor by factor** — the product is auditable, "
                 "not asserted")
        st.dataframe(pd.DataFrame(per_ch)[
            ["channel", "model_id", "reliability", "extremeness",
             "counterfactual", "stability", "corroboration", "signed_evidence",
             "solidity"]], use_container_width=True)

    sh = audit.get("shapley") or {}
    if sh:
        import pandas as pd
        st.write(f"**Exact branch Shapley** — {sh.get('n_forwards', 0)} coalitions, "
                 f"efficiency residual {sh.get('efficiency_residual', float('nan')):.2e}")
        st.dataframe(pd.DataFrame([
            {"branch": k, "shapley (logits)": val,
             "share": (sh.get("normalised") or {}).get(k, float("nan"))}
            for k, val in (sh.get("values") or {}).items()]),
            use_container_width=True)
        inter = sh.get("interactions") or {}
        if inter:
            st.caption("pairwise interactions: " + ", ".join(
                f"{k} {val:+.4f}" for k, val in sorted(
                    inter.items(), key=lambda kv: -abs(kv[1]))[:3]))
        for n in sh.get("notes", []):
            st.warning(n)

    cands = audit.get("candidates") or []
    if cands:
        import pandas as pd
        st.write("**Every candidate explanation, and whether it earned its place**")
        st.dataframe(pd.DataFrame([
            {"method": c["method"], "modality": c["modality"],
             "grade": c["grade"], "admissible": c["admissible"],
             "beats random": c["beats_random"],
             "margin over random": c["margin_over_random"],
             "solidity": c["solidity"],
             "why not": "; ".join(c.get("notes") or [])} for c in cands]),
            use_container_width=True)

    nar = audit.get("narration") or {}
    if nar:
        st.write("**Narration checks**")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("backend", nar.get("backend", "?"))
        n2.metric("groundedness", f"{nar.get('groundedness', 0):.2f}")
        n3.metric("FK grade", f"{nar.get('fk_grade', 0):.1f}")
        n4.metric("numerals clean",
                  "yes" if (nar.get("numeral_check") or {}).get("ok") else "NO")
        if nar.get("dropped_claims"):
            st.write("Sentences dropped before display:")
            for d in nar["dropped_claims"]:
                st.caption(f"- \"{d['sentence']}\" — {d['reason']}")
        for n in nar.get("notes", []):
            st.caption(f"- {n}")

    for n in audit.get("notes", []):
        st.warning(n)
    st.download_button("download this card as JSON",
                       json.dumps(audit, indent=2, default=str),
                       file_name="cfx_card.json", mime="application/json")


# ==========================================================================
# page
# ==========================================================================

st.title("FSV-Net — verdict, confidence, and one explanation that earned its place")

with st.sidebar:
    st.header("Reading")
    register = st.radio("Explanation style", ["Plain language", "Technical"],
                        index=0,
                        help="Switches between two texts that were both written "
                             "when the clip was analysed. Both describe the same "
                             "evidence; only the wording differs.")
    reg = "layman" if register == "Plain language" else "technical"
    st.divider()
    st.header("Honest limits")
    st.caption(
        "This detector's genuinely held-out accuracy on unseen identities is "
        "close to chance, and its full five-branch configuration measured "
        "*worse* than its own three-branch ablation. So it abstains often, by "
        "design. An abstention here is the system working, not failing.")
    st.divider()
    cpu = st.checkbox("force CPU", value=False)

saved = load_saved_cards()
tab_browse, tab_run = st.tabs(["Browse analysed clips", "Analyse a clip"])

with tab_browse:
    if not saved:
        st.info("No saved cards yet. Run `python scripts/cfx_eval.py --n 20` to "
                "produce some, or use the other tab.")
    else:
        pick = st.selectbox(f"{len(saved)} analysed clips", sorted(saved))
        card = saved[pick]
        v = card["verdict"]
        nar = card.get("narration") or {}
        user = {
            "decision": v["decision"],
            "confidence_number": round(v["p_fake_calibrated"], 3),
            "confidence_band": [round(v["band"][0], 3), round(v["band"][1], 3)],
            "text": (nar.get(reg) or ""),
            "explanation_available": card.get("best") is not None,
            "explanation_of": ((card.get("best") or {}).get("statement")
                               or card.get("withheld_reason") or ""),
            "figure": (card.get("figures") or {}).get("primary", ""),
        }
        render_user_surface(user, reg)
        st.divider()
        with st.expander("Audit — how this answer was produced", expanded=False):
            render_audit(card)

with tab_run:
    pipe, device, warn = load_pipeline(cpu)
    if pipe is None:
        st.error(warn)
    else:
        st.caption(f"device: {device}" + (f"  —  {warn}" if warn else ""))
        if warn:
            st.warning(warn + ". Numbers shown before that script has run are "
                       "uncalibrated and nothing will be abstained on.")
        st.write("Pick a clip from the Celeb-DF-v2 frame directory:")
        root = ROOT / "Celeb-DF-v2"
        subs = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
        if not subs:
            st.error(f"no Celeb-DF-v2 directory at {root}")
        else:
            which = st.selectbox("folder", subs)
            clips = sorted(p.name for p in (root / which).iterdir() if p.is_dir())
            name = st.selectbox(f"{len(clips)} clips", clips)
            if st.button("Analyse", type="primary"):
                from fsvnet import celebdf_frames as cf
                from fsvnet.dataset_frames import normalise
                from fsvnet.explain import denormalise
                with st.spinner("analysing — the expensive explainers only run "
                                "if the verdict clears the confidence gate"):
                    frames, _lm, _meta = cf.load_clip(
                        str(root / which / name), None, n_frames=16,
                        img_size=224, mode="uniform")
                    clip = normalise(frames).unsqueeze(0).to(device)
                    card = pipe.analyse(clip, frames_u8=denormalise(clip[0]),
                                        clip_id=name)
                render_user_surface(card.for_user(reg), reg)
                st.divider()
                with st.expander("Audit — how this answer was produced",
                                 expanded=False):
                    render_audit(card.for_audit())
