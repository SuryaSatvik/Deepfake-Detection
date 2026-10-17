"""
narrate.py
==========

One evidence card, two registers: technical and layman. The UI toggles between
them; the LLM is never called by the toggle.

The rule the whole module hangs on
----------------------------------
    **The toggle changes the register, never the content.**

If the technical and layman texts can disagree about a fact, the system is lying
to one of the two audiences, and it will be the non-technical one. So both
registers are rendered from the same `ExplanationCard`, both are generated at
analysis time, both are cached on the card, and the toggle is a switch between
two strings. This also solves the 8 GB problem -- Qwen is loaded once per clip,
not once per click -- and makes the demo instant.

Why the layman register is the highest-risk output in this project
-----------------------------------------------------------------
A specialist reading "moderate evidence, p=0.71, identity-disjoint AUC 0.57"
discounts it correctly. A non-specialist reading fluent prose about a suspicious
video does not. So the layman path is the most constrained thing here:

1. **Controlled lexicon** (`assets/lexicon.yaml`) -- a measurement may only be
   described using the phrase the lexicon gives it. The model cannot invent a
   folksy name for something it did not measure.
2. **Claim linking** -- the model must return JSON: each sentence with the
   evidence ids it rests on. A sentence with no ids is *dropped*, not rewritten.
3. **Fixed confidence phrasing** -- the same probability always yields the same
   words, plus a natural-frequency framing ("about 7 times in 10"), plus the
   number itself. Never the words instead of the number.
4. **Stricter forbidden lexicon** than the technical register, and a hard reject
   for any sentence mentioning something FSV-Net cannot measure (blinking, lip
   sync, lighting -- the things a VLM reaches for when it has nothing).

And the technical register gets its own guard: `numeral_fidelity` extracts every
number from the generated text and checks it against the card. A model that
rounds 0.573 to "about 0.6" has changed a measurement, and that is caught
mechanically rather than by reading.

Everything degrades to a template. No sentence in this system requires a model to
be loaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .utils import log
except Exception:                                     # pragma: no cover
    import logging
    log = logging.getLogger("fsvnet")

TECHNICAL = "technical"
LAYMAN = "layman"
REGISTERS = (TECHNICAL, LAYMAN)

def _default_lexicon_paths() -> List[Path]:
    """
    Walk up from this module looking for assets/lexicon.yaml.

    In the repo that resolves to <root>/assets/lexicon.yaml from
    src/fsvnet/narrate.py, but walking rather than hard-coding a parent index
    means the package still finds its lexicon when it is installed, vendored, or
    copied into a test tree.
    """
    here = Path(__file__).resolve()
    cands = [p / "assets" / "lexicon.yaml" for p in here.parents[:5]]
    cands.append(here.parent / "lexicon.yaml")
    cands.append(Path.cwd() / "assets" / "lexicon.yaml")
    return cands


# ==========================================================================
# lexicon
# ==========================================================================


@dataclass
class Lexicon:
    terms: Dict[str, Dict[str, str]] = field(default_factory=dict)
    confidence: List[Dict[str, object]] = field(default_factory=list)
    abstain: Dict[str, object] = field(default_factory=dict)
    forbidden: Dict[str, List[str]] = field(default_factory=dict)
    ungrounded_terms: List[str] = field(default_factory=list)
    readability: Dict[str, float] = field(default_factory=dict)
    version: int = 1

    # ------------------------------------------------------------------
    @staticmethod
    def load(path=None) -> "Lexicon":
        import yaml
        cands = [Path(path)] if path else _default_lexicon_paths()
        for p in cands:
            if p.exists():
                d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                return Lexicon(terms=d.get("terms", {}),
                               confidence=d.get("confidence", []),
                               abstain=d.get("abstain", {}),
                               forbidden=d.get("forbidden", {}),
                               ungrounded_terms=d.get("ungrounded_terms", []),
                               readability=d.get("readability", {}),
                               version=int(d.get("version", 1)))
        raise FileNotFoundError(
            "lexicon.yaml not found. The layman register is defined by that file, "
            "so it is a required asset rather than an optional one. Looked in: "
            + ", ".join(str(c) for c in cands))

    # ------------------------------------------------------------------
    def phrase(self, term: str, register: str = LAYMAN) -> str:
        e = self.terms.get(term)
        if not e:
            return term.replace("_", " ")
        return str(e.get(register, e.get(TECHNICAL, term)))

    def reverse_lookup(self, text: str) -> List[str]:
        """Which evidence terms a piece of layman text actually refers to."""
        t = text.lower()
        hits: List[str] = []
        for term, e in self.terms.items():
            lay = str(e.get(LAYMAN, "")).lower()
            # match on the distinctive opening of the phrase, not the whole
            # sentence, so light paraphrase still resolves
            key = " ".join(lay.split()[:5])
            if (key and key in t) or term.replace("_", " ") in t or term in t:
                hits.append(term)
        return hits

    def confidence_entry(self, p: float) -> Dict[str, object]:
        for e in sorted(self.confidence, key=lambda x: float(x["max"])):
            if p < float(e["max"]):
                return e
        return self.confidence[-1] if self.confidence else {
            "layman": "unclear", "technical": "unclear", "verdict": "uncertain"}


# ==========================================================================
# confidence phrasing
# ==========================================================================


def natural_frequency(p: float) -> str:
    """
    '0.71' -> 'about 7 times in 10'.

    People read natural frequencies far more accurately than probabilities, and
    the phrasing makes the error rate impossible to ignore: "7 times in 10" says
    "wrong 3 times in 10" at the same moment.
    """
    p = float(min(max(p, 0.0), 1.0))
    n = int(round(p * 10))
    if n <= 0:
        return "fewer than 1 time in 10"
    if n >= 10:
        return "more than 9 times in 10"
    return f"about {n} time{'s' if n != 1 else ''} in 10"


def confidence_phrase(p: float, decision: str, lex: Lexicon,
                      register: str = LAYMAN) -> str:
    if decision == "abstain":
        return str(lex.abstain.get("headline", "I can't tell."))
    e = lex.confidence_entry(p)
    words = str(e.get(register, e.get(LAYMAN, "unclear")))
    if register == LAYMAN:
        return f"{words} ({natural_frequency(p)} the detector is right at this score)"
    return f"{words} (p(fake) = {p:.3f})"


# ==========================================================================
# checks
# ==========================================================================

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numerals(text: str) -> List[str]:
    return _NUM.findall(text or "")


def numeral_fidelity(text: str, allowed: Sequence[float],
                     tol: float = 5e-4) -> Dict[str, object]:
    """
    Every number in a technical rendering must come from the card.

    This catches the commonest quiet failure of an LLM-smoothed template: the
    model restates 0.573 as "roughly 0.6", or invents a percentage. Rounding a
    measurement is changing it.
    """
    found = extract_numerals(text)
    allowed_f = [float(a) for a in allowed]
    bad: List[str] = []
    for f in found:
        v = float(f)
        if not any(abs(v - a) <= max(tol, abs(a) * 1e-3) for a in allowed_f):
            bad.append(f)
    return {"ok": not bad, "unsupported": bad, "n_found": len(found)}


def forbidden_check(text: str, words: Sequence[str]) -> List[str]:
    t = (text or "").lower()
    return [w for w in words if w.lower() in t]


def ungrounded_check(text: str, terms: Sequence[str]) -> List[str]:
    t = (text or "").lower()
    return [w for w in terms if w.lower() in t]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def flesch_kincaid_grade(text: str, use_textstat: bool = False) -> float:
    """
    FK grade level, so "written for non-technical readers" is a measurement
    rather than a claim.

    The local implementation is the default on purpose: it has no dependencies
    and no network access. `textstat` is slightly more accurate on unusual words
    but reaches for an NLTK corpus download on first use, which fails noisily
    behind a proxy and would spam the log on every clip. Pass
    `use_textstat=True` if it is installed and its corpora are already present.
    """
    if use_textstat:
        try:
            import textstat
            return float(textstat.flesch_kincaid_grade(text))
        except Exception:
            pass
    words = re.findall(r"[A-Za-z']+", text or "")
    sents = max(1, len(split_sentences(text)))
    if not words:
        return 0.0

    def syll(w: str) -> int:
        w = w.lower()
        groups = re.findall(r"[aeiouy]+", w)
        n = len(groups)
        if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
            n -= 1
        return max(1, n)

    sy = sum(syll(w) for w in words)
    return float(0.39 * (len(words) / sents) + 11.8 * (sy / len(words)) - 15.59)


# ==========================================================================
# claim linking
# ==========================================================================


@dataclass
class Claim:
    sentence: str
    supports: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sentence": self.sentence, "supports": list(self.supports)}


def parse_claim_json(raw: str) -> Tuple[List[Claim], Optional[str]]:
    """
    Parse the model's JSON. Returns (claims, error). Tolerant of code fences and
    of leading prose, because a 3B model will sometimes wrap its JSON in both.
    """
    if not raw:
        return [], "empty response"
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE).strip()
    m = re.search(r"(\[.*\]|\{.*\})", txt, flags=re.DOTALL)
    if not m:
        return [], "no JSON found in response"
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return [], f"invalid JSON: {e}"
    if isinstance(data, dict):
        data = data.get("claims") or data.get("sentences") or []
    out: List[Claim] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            s = str(item.get("sentence", "")).strip()
            sup = item.get("supports") or item.get("evidence") or []
            if isinstance(sup, str):
                sup = [sup]
            if s:
                out.append(Claim(s, [str(x) for x in sup]))
        elif isinstance(item, str) and item.strip():
            out.append(Claim(item.strip(), []))
    return out, None if out else "no usable sentences"


def claim_link_filter(claims: Sequence[Claim], valid_ids: Sequence[str],
                      lex: Optional[Lexicon] = None
                      ) -> Tuple[List[Claim], List[Dict[str, object]]]:
    """
    Drop every sentence that does not rest on a surviving piece of evidence.

    Dropped, not rewritten. Rewriting an unsupported sentence produces a
    supported-looking sentence that no measurement backs; deleting it loses
    nothing that was ever true.
    """
    valid = set(valid_ids)
    kept: List[Claim] = []
    dropped: List[Dict[str, object]] = []
    for c in claims:
        sup = [s for s in c.supports if s in valid]
        reason = None
        if not c.supports:
            reason = "no evidence id cited"
        elif not sup:
            reason = (f"cites evidence not in the surviving set "
                      f"({', '.join(c.supports)})")
        elif lex is not None:
            bad = ungrounded_check(c.sentence, lex.ungrounded_terms)
            if bad:
                reason = f"mentions unmeasurable properties: {', '.join(bad)}"
        if reason:
            dropped.append({"sentence": c.sentence, "reason": reason})
        else:
            kept.append(Claim(c.sentence, sup))
    return kept, dropped


# ==========================================================================
# prompts
# ==========================================================================

LAYMAN_SYSTEM = """You explain a forensic video detector's output to someone with
no technical background. You are given a list of measurements, each with an id, a
plain-language description, and how unusual it is compared with real videos.

Write 2 to 4 short sentences. Rules, all of them hard:
- Use ONLY the plain-language descriptions you are given. Do not invent your own
  way of describing a measurement, and do not mention anything not listed.
- Every sentence must rest on at least one listed measurement id.
- Never claim certainty. Never use the words clearly, obviously, definitely,
  certainly, proof, confirmed, or 100%.
- Never mention blinking, eye movement, lip sync, shadows, reflections, teeth,
  lighting or audio. The detector does not measure any of them.
- Do not state a verdict or a probability. That sentence is added separately.
- Short, everyday words. Aim for the reading level of a newspaper front page.

Reply with JSON only, in this exact shape:
[{"sentence": "...", "supports": ["evidence_id", ...]}, ...]"""

TECHNICAL_SYSTEM = """You are a forensic analyst writing for a technical reader.
You are given a detector's measurements and attribution results.

Rules:
- Use only the numbers supplied. Do not round them, restate them approximately,
  or introduce any number that is not in the input.
- Do not claim certainty; report what was measured and what it bounds.
- At most 150 words, prose, no headers or bullets.

Your job is to join the supplied facts into readable prose. You are not adding
analysis."""


def build_layman_prompt(evidence_items: Sequence[Dict[str, object]],
                        lex: Lexicon) -> str:
    """
    Render the surviving evidence for the layman pass. Only items that passed the
    faithfulness and solidity gates should be here -- what the model cannot see,
    it cannot narrate.
    """
    lines = ["Measurements available to you:"]
    for e in evidence_items:
        eid = e["id"]
        desc = e.get("layman") or lex.phrase(str(e.get("term", eid)), LAYMAN)
        how = e.get("how_unusual", "")
        lines.append(f"- id: {eid}")
        lines.append(f"  means: {desc}")
        if how:
            lines.append(f"  how unusual: {how}")
    lines.append("")
    lines.append("Write the JSON now. 2-4 sentences, each citing at least one id.")
    return "\n".join(lines)


# ==========================================================================
# deterministic renderings (the fallback, and the floor)
# ==========================================================================


def layman_template(evidence_items: Sequence[Dict[str, object]],
                    lex: Lexicon) -> str:
    """
    Plain-language text with no model involved. Duller than a generated version
    and impossible to get wrong -- every clause is a direct rendering of a
    measurement that already passed the gates.
    """
    if not evidence_items:
        return ("Nothing the detector measures stood out on this clip, so there "
                "is nothing to point at as a reason.")
    bits = []
    for e in list(evidence_items)[:3]:
        desc = e.get("layman") or lex.phrase(str(e.get("term", e["id"])), LAYMAN)
        how = e.get("how_unusual")
        bits.append(f"{desc}" + (f" ({how})" if how else ""))
    if len(bits) == 1:
        return f"One thing stood out: {bits[0]}."
    head = "; ".join(bits[:-1])
    return f"A few things stood out: {head}; and {bits[-1]}."


def abstain_text(lex: Lexicon, register: str = LAYMAN,
                 include_help: bool = True) -> str:
    """
    The most frequently shown string in the system, at this detector's honest
    accuracy. Templated rather than generated, and written to say "I cannot tell"
    in the first sentence without apologising its way into sounding like a soft
    yes.
    """
    a = lex.abstain
    parts = [str(a.get("headline", "I can't tell whether this video is genuine."))]
    if register == LAYMAN:
        parts.append(str(a.get("body", "")).strip())
        if include_help and a.get("what_would_help"):
            parts.append("What would help: "
                         + "; ".join(str(x) for x in a["what_would_help"]) + ".")
    else:
        parts.append("The verdict falls inside the fitted abstention band, so no "
                     "class is reported. Abstention is the designed behaviour "
                     "here, not a failure of the run.")
    return " ".join(p for p in parts if p)


# ==========================================================================
# the narration object
# ==========================================================================


@dataclass
class Narration:
    technical: str
    layman: str
    backend: str = "template"
    claims: List[Dict[str, object]] = field(default_factory=list)
    dropped_claims: List[Dict[str, object]] = field(default_factory=list)
    groundedness: float = 0.0
    fk_grade: float = 0.0
    numeral_check: Dict[str, object] = field(default_factory=dict)
    back_translation_ok: bool = True
    fell_back_to_template: bool = False
    notes: List[str] = field(default_factory=list)

    def text(self, register: str = LAYMAN) -> str:
        """The toggle. A pure switch between two cached strings -- no model call."""
        if register not in REGISTERS:
            raise ValueError(f"register must be one of {REGISTERS}, got {register!r}")
        return self.technical if register == TECHNICAL else self.layman

    def to_dict(self) -> dict:
        return asdict(self)


def back_translation_check(layman_text: str, expected_ids: Sequence[str],
                           id_to_term: Dict[str, str], lex: Lexicon
                           ) -> Dict[str, object]:
    """
    Map the finished layman text back to evidence terms and compare with what it
    was supposed to rest on.

    A cheap, deterministic second opinion: if the text no longer refers to the
    evidence it cited, something drifted during generation. Divergence triggers
    the template fallback, and the pass rate over a test subsample is the
    reportable **layman groundedness rate** -- a measurement rather than a claim.
    """
    found = set(lex.reverse_lookup(layman_text))
    expected = {id_to_term.get(i, i) for i in expected_ids}
    if not expected:
        return {"ok": True, "found": sorted(found), "expected": [],
                "recovered": 1.0}
    rec = len(found & expected) / len(expected)
    return {"ok": bool(rec >= 0.5), "found": sorted(found),
            "expected": sorted(expected), "recovered": float(rec)}


def narrate(evidence_items: Sequence[Dict[str, object]],
            technical_text: str,
            allowed_numerals: Sequence[float],
            decision: str,
            p_calibrated: float,
            lex: Optional[Lexicon] = None,
            reasoner=None, frames=None, max_retries: int = 1) -> Narration:
    """
    Produce both registers, once, for one clip.

    `reasoner` is an `EvidenceReasoner`-like object with `.load()` and a backend
    exposing `.generate(system, user, images)`. When it is None, unavailable, or
    fails any check, the layman register comes from the template. Nothing in this
    function can fail in a way that leaves a user without text.

    `technical_text` is built deterministically by the caller (cfx.py) so the
    numbers are guaranteed correct; the model is only ever asked for the layman
    register here.
    """
    lex = lex or Lexicon.load()
    notes: List[str] = []

    conf_lay = confidence_phrase(p_calibrated, decision, lex, LAYMAN)
    conf_tech = confidence_phrase(p_calibrated, decision, lex, TECHNICAL)

    # ---- abstain: templated in both registers, no model involved -------
    if decision == "abstain":
        return Narration(
            technical=f"{abstain_text(lex, TECHNICAL)} {technical_text}".strip(),
            layman=abstain_text(lex, LAYMAN), backend="template",
            groundedness=1.0, fk_grade=flesch_kincaid_grade(abstain_text(lex, LAYMAN)),
            numeral_check={"ok": True, "unsupported": [], "n_found": 0},
            fell_back_to_template=True,
            notes=["verdict inside the abstention band: both registers are "
                   "templated, which is deliberate -- this is the sentence the "
                   "system says most often and it should not vary run to run."])

    id_to_term = {str(e["id"]): str(e.get("term", e["id"])) for e in evidence_items}
    valid_ids = list(id_to_term.keys())

    tech = f"{conf_tech} {technical_text}".strip()
    num = numeral_fidelity(tech, list(allowed_numerals) + [p_calibrated])
    if not num["ok"]:
        notes.append(f"technical register contains numerals not present in the "
                     f"card: {num['unsupported']}. This is a bug in the template, "
                     "not a style issue -- fix it before reporting.")

    # ---- layman: generate, check, fall back ----------------------------
    lay: Optional[str] = None
    claims: List[Claim] = []
    dropped: List[Dict[str, object]] = []
    backend = "template"
    bt = {"ok": True, "recovered": 1.0, "found": [], "expected": []}

    if reasoner is not None and evidence_items:
        prompt = build_layman_prompt(evidence_items, lex)
        for attempt in range(max_retries + 1):
            try:
                be = reasoner.load()
                if be is None or be.__class__.__name__ == "_TemplateBackend":
                    notes.append("no generative backend available; layman register "
                                 "is templated.")
                    break
                raw = be.generate(LAYMAN_SYSTEM, prompt, list(frames or []))
                parsed, err = parse_claim_json(raw)
                if err:
                    notes.append(f"attempt {attempt + 1}: {err}")
                    continue
                kept, drop = claim_link_filter(parsed, valid_ids, lex)
                dropped = drop
                if not kept:
                    notes.append(f"attempt {attempt + 1}: every sentence was "
                                 "dropped by the claim-link filter.")
                    continue
                cand = " ".join(c.sentence for c in kept)
                bad = forbidden_check(cand, lex.forbidden.get(LAYMAN, []))
                if bad:
                    notes.append(f"attempt {attempt + 1}: overconfident wording "
                                 f"{bad} -- rejected.")
                    continue
                cited = sorted({s for c in kept for s in c.supports})
                bt = back_translation_check(cand, cited, id_to_term, lex)
                if not bt["ok"]:
                    notes.append(f"attempt {attempt + 1}: back-translation "
                                 f"recovered only {bt['recovered']:.0%} of the "
                                 "cited evidence -- rejected.")
                    continue
                fk = flesch_kincaid_grade(cand)
                max_fk = float(lex.readability.get("max_fk_grade", 10.0))
                if fk > max_fk:
                    notes.append(f"attempt {attempt + 1}: reading level FK {fk:.1f} "
                                 f"above the {max_fk:.0f} ceiling -- rejected.")
                    continue
                lay, claims = cand, kept
                backend = getattr(be, "mode", "vlm")
                break
            except Exception as e:                       # never fail the pipeline
                notes.append(f"attempt {attempt + 1}: generation error "
                             f"{e.__class__.__name__}: {e}")

    fell_back = lay is None
    if fell_back:
        lay = layman_template(evidence_items, lex)
        backend = "template"

    lay_full = f"{conf_lay.capitalize()}. {lay}".replace("..", ".")
    grounded = (len(claims) / max(1, len(claims) + len(dropped))) if claims else (
        1.0 if fell_back else 0.0)

    return Narration(
        technical=tech, layman=lay_full, backend=backend,
        claims=[c.to_dict() for c in claims], dropped_claims=dropped,
        groundedness=float(grounded), fk_grade=flesch_kincaid_grade(lay_full),
        numeral_check=num, back_translation_ok=bool(bt["ok"]),
        fell_back_to_template=fell_back, notes=notes)


__all__ = ["TECHNICAL", "LAYMAN", "REGISTERS", "Lexicon", "natural_frequency",
           "confidence_phrase", "extract_numerals", "numeral_fidelity",
           "forbidden_check", "ungrounded_check", "split_sentences",
           "flesch_kincaid_grade", "Claim", "parse_claim_json",
           "claim_link_filter", "LAYMAN_SYSTEM", "TECHNICAL_SYSTEM",
           "build_layman_prompt", "layman_template", "abstain_text",
           "Narration", "back_translation_check", "narrate"]
