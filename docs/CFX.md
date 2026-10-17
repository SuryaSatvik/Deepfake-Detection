# CFX — Calibrated & Faithful eXplanation

The explanation layer for FSV-Net: a verdict with a calibrated confidence, one
explanation that has measurably earned its place, and two reading registers for
the same evidence.

Added September 2026. Design rationale in the project doc
`claude/xai_confidence_design.md`; this file is the implementation reference.

---

## 1. Why it is shaped this way

| measurement | value | source |
|---|---:|---|
| identity-disjoint channel probes | 0.51 – 0.59 AUC | `results/comparison/branch_probe_comparison.json` |
| full model − conventional (B0) | **−0.111 AUC**, resolved, p≈0 | `results/ablations.csv` |
| ECE, full model | 0.081 (equal-width) | `results/calibration.csv` |
| class prior | 86.4% fake | `docs/FINDINGS.md` |

A detector at that level cannot honestly support a layer that says "here is why
this video is fake". So CFX is built around the inverse claim, which the data
does support:

> **The explanation layer's job is to decide when the model has earned the right
> to explain itself, and to refuse when it has not.**

Two numbers come out, and they are never merged:

- **verdict confidence** — a calibrated probability plus an abstention decision
- **explanation faithfulness** — how much of the decision the shown explanation
  actually accounts for

A system can be confident and unexplainable, or unconfident with a perfectly
faithful explanation. Collapsing those into one score destroys the only
interesting information in them.

---

## 2. Modules

| file | what it owns |
|---|---|
| `calibrate.py` | temperature / Platt scaling, adaptive + equal-width ECE, MCE, Brier decomposition, `reprior()` |
| `uncertainty.py` | window spread, TTA (incl. a real JPEG round-trip), MC dropout |
| `selective.py` | Mondrian split-conformal abstention, risk–coverage, AURC/E-AURC, `assert_coverage_sane` |
| `attribution.py` | exact branch Shapley, integrated gradients, KernelSHAP over superpixels, three counterfactuals |
| `faithfulness.py` | deletion / insertion / comprehensiveness / sufficiency, dual fills, random + blur controls, Borda, stability |
| `arbitrate.py` | the reliability matrix, per-clip solidity, routing, the fitted confidence stack |
| `narrate.py` | the two registers, the controlled lexicon, claim linking, numeral fidelity, readability |
| `cfx.py` | `CFXPipeline`, `ExplanationCard`, the four-quadrant policy, the user/audit boundary |
| `assets/lexicon.yaml` | every plain-language phrase the system is allowed to use |

Reused, not replaced: `explain.py` (its four methods become candidates),
`evidence.ReferenceStats`, `benchmark.paired_bootstrap`,
`model.forward(ablate_branches=…)`, `vlm_reasoner`.

`config.py` is untouched. CFX's tunables live in `cfx.CFXConfig`, passed to the
pipeline — `config.py` is the single source of truth for the detector and the
training run, and this layer is downstream of both.

---

## 3. Running it

```bash
python scripts/test_cfx.py                  # 142 checks, no checkpoints needed
python scripts/cfx_fit.py --limit 32        # smoke: does everything load?
python scripts/cfx_fit.py --alpha 0.10      # fit on val, write the artifacts
python scripts/cfx_eval.py --n 8            # smoke
python scripts/cfx_eval.py --n 200 --narrate
streamlit run streamlit/cfx_app.py
```

`notebooks/CFX_RUN_ALL.ipynb` drives all of it and prints the report tables.

The design doc named four scripts; they shipped as two, because the calibration /
policy / reliability / stack stages all need the same expensive scoring pass and
splitting them would mean scoring four checkpoints twice.

### Artifacts, all under `results/cfx/`

```
scores_<id>.npz            y / p / fake-margin per split, + per-channel ablation deltas
calibration_<id>.json      chosen calibrator, CV scores, before/after reports
abstention_policy.json     the fitted band (val) — applied unchanged to test
evidence_reliability.json  the (model × channel) matrix
confidence_stack.json      the arbitration weights
cfx_*.csv                  the report tables
cards/<clip_id>.json       one ExplanationCard per analysed clip
```

---

## 4. The specialist panel

The four condition checkpoints are not an ensemble — they differ in which
branches exist, not in random seed, so their mean has no interpretation. They are
a panel of **specialists**, each authoritative for the channels it carries:

| id | `(use_tsc, use_lmf)` | channels | role |
|---|---|---|---|
| `B0` | (F, F) | spatial, frequency, srm | conventional — **best honest AUC, the baseline to beat** |
| `C1` | (T, F) | + tsc | temporal-spectral specialist |
| `C2` | (F, T) | + lmf | landmark-motion specialist |
| `C3` | (T, T) | all five | full (PROPOSED) — −0.111 AUC vs B0 |

These are the same ids `scripts/train_conditions.py` uses and the same ones
carried inside `models/checkpoints/fsvnet_dfe_<id>.pt`. They are deliberately not
renamed: a second naming scheme for the same four checkpoints is how a routing
table ends up pointing at the wrong model.

### Solidity

Per clip, per channel:

```
solidity = reliability × extremeness × counterfactual × stability × corroboration
```

Every factor is in [0,1] and every factor is **stored separately**, so the product
can be audited line by line rather than taken on faith. A zero in any factor
zeroes the channel, by design:

- `reliability` — from the matrix. A channel that failed its random-attribution
  control scores 0 outright.
- `extremeness` — where the measurement sits in the real-video reference
  distribution, **with a ±10-percentile deadband**. Anything from the 40th to the
  60th percentile scores exactly zero, not a small number. A small-but-positive
  weight is precisely what lets a middling measurement accumulate into a
  confident story.
- `counterfactual` — did removing this channel move the logit at all.
- `stability` — does the explanation survive reseeding and σ=0.01 noise.
- `corroboration` — do the specialists carrying this channel agree on its sign
  and rough magnitude. Disagreement is a red flag, not an average to take.

### The confidence stack

An L2 logistic regression over the solidity-weighted evidence, fitted on val,
then temperature-calibrated. The coefficients **are** the arbitration weights:
learned, printable, falsifiable. There is no hand-written weighted sum anywhere
in `arbitrate.py`.

**The baseline it must beat** is B0 with plain temperature scaling, on a paired
bootstrap over the same clips. `compare_to_single_model()` returns an explicit
"ship the baseline" recommendation when it does not win. Report that outcome if
it happens — the stack does not earn its complexity by existing.

### A prediction, written down in advance

Given `results/ablations.csv`, arbitration will most likely route most weight to
**B0** and score TSC and LMF low. That would be the dataset-level ablation
finding rediscovered automatically, per clip. It is written here *before* the run
so that "it came out as expected" is a check rather than a story told afterwards.

---

## 5. Faithfulness

Every candidate is reduced to a ranking over input units — patches, frames,
branches, superpixels — and then scored by acting on it:

| metric | direction |
|---|---|
| deletion AUC | lower better |
| insertion AUC | higher better |
| comprehensiveness | higher better |
| sufficiency | closer to zero better |

Aggregated by **Borda count**, not a weighted sum: any weighting would be an
arbitrary choice presented as a measurement, and it is the first thing a reviewer
would attack.

Three mandatory controls:

1. **Random attribution.** The same units in a random order, scored identically.
   A candidate that does not beat it is not an explanation and is not shown —
   whatever the picture looks like. This is the single most important number in
   the module and the one most often missing from published XAI results.
2. **Blur-everything.** How far the margin moves when all fine detail goes, with
   no attribution involved. It bounds how much of any deletion curve is evidence
   removal rather than image degradation.
3. **Two fills.** Blurred *and* an in-distribution patch from another frame of
   the same clip. Black or blurred fill both create inputs the model never saw,
   so part of the drop is an out-of-distribution reaction. Disagreement between
   fills is flagged, and neither number is reported alone.

### The modality gate

Exact branch Shapley says where the decision came from, and that decides which
explanation families are admissible at all:

> If spatial contributed 12% of the decision, a Grad-CAM map explains 12% of the
> decision, and presenting it as "the reason" is a misrepresentation — however
> well it scores on deletion.

Ordering: Shapley → admissible modalities → faithfulness within that set →
stability filter → winner. That ordering is the design's actual contribution, and
it exists only because the model is multi-branch.

### Why exact Shapley rather than single ablation

`explain.branch_attribution` measures `v(all) − v(all \ i)`, which equals the
Shapley value only when branches do not interact. Redundant branches are exactly
the case where they do: single ablation makes both look unimportant. With K ≤ 5
there are only 32 coalitions, so the exact value is a closed enumeration — 32
forward passes, no sampling. Pairwise interaction indices come out of the same
enumeration for free.

The efficiency residual (`Σφᵢ` vs `v(N) − v(∅)`) is **returned, not asserted**: a
large residual means the forward pass is not deterministic — dropout left on,
non-deterministic kernels — and every number should be distrusted until that is
fixed.

---

## 6. The two registers

**The toggle changes the register, never the content.** One `ExplanationCard`,
both texts generated at analysis time and cached on it, and the UI toggle is a
dictionary lookup. If the two registers could disagree about a fact, the system
would be lying to one of the two audiences — and it would be the non-technical
one. It also means Qwen is loaded once per clip rather than once per click, which
is what makes it fit an 8 GB card.

The layman register is the highest-risk output in this project. A specialist
reading "moderate evidence, p=0.71" discounts it correctly; a non-specialist
reading fluent prose about a suspicious video does not. So it is the most
constrained path in the codebase:

1. **Controlled lexicon** (`assets/lexicon.yaml`) — a measurement may only be
   described with the phrase the lexicon gives it.
2. **Claim linking** — the model returns JSON, one entry per sentence with the
   evidence ids it rests on. A sentence with no surviving id is **dropped, not
   rewritten**: rewriting produces a supported-looking sentence that nothing
   backs; deleting loses nothing that was ever true.
3. **Fixed confidence phrasing** — the same probability always yields the same
   words, plus a natural-frequency framing ("about 7 times in 10", which also
   says "wrong 3 times in 10"), plus the number itself. The words are *in
   addition to* the number, never instead of it.
4. **Stricter forbidden lexicon** than the technical register, and a hard reject
   for anything FSV-Net cannot measure — blinking, lip sync, lighting, shadows:
   the things a VLM reaches for when it has nothing.
5. **Back-translation** — map the finished text back to evidence terms and
   compare with what it cited. Divergence falls back to the template, and the
   pass rate is the reportable **layman groundedness rate**.

The technical register has its own guard: `numeral_fidelity` extracts every
number from the text and checks it against the card. A model that restates 0.573
as "about 0.6" has changed a measurement, and that is caught mechanically.

Everything degrades to a deterministic template. No sentence in this system
requires a model to be loaded.

### The abstain message

At this detector's honest accuracy it is the **most frequently shown string in
the system**, so it is templated rather than generated, written once, and leads
with "I can't tell" in the first sentence without apologising its way into
sounding like a soft yes. It is in `lexicon.yaml` under `abstain`, and it is the
sentence to put on the demo slide.

---

## 7. The output policy and the abstraction boundary

| verdict confidence | explanation faithfulness | output |
|---|---|---|
| high | high | verdict + explanation + both scores |
| high | low | verdict + "no reliable explanation available" |
| low | high | no verdict; evidence as an observation, not a conclusion |
| low | low | abstain entirely |

`card.for_user(register)` returns the verdict, the confidence **and its band**,
one explanation, one figure. `card.for_audit()` returns everything else. The
audit panel is collapsed in the UI, not removed — the examiner *is* the technical
audience, and "the interesting parts are one click away" demos better than "trust
me".

Two things the UI must never do: expose model selection as a setting (arbitration
is the contribution — a dropdown gives it away), and show a confidence number
without its band.

Expensive candidates and the Qwen pass run **only** when the verdict clears the
confidence gate. In the stub tests that is 14 forwards for an abstained clip
versus 330 for an explained one.

---

## 8. Compute

RTX 4060, 8 GB, T=16 frames:

| stage | forwards |
|---|---:|
| exact branch Shapley | 32 |
| integrated gradients, 32 steps × 2 baselines | 64 |
| KernelSHAP, 2 frames × 512 coalitions | ~1,024 |
| deletion / insertion / comp / suff, per method × 2 fills | ~280 |
| counterfactuals (spectral, temporal, region) | ~150 |
| stability repeats | ~300 |
| specialist panel | 4 |

≈ 1,900–2,800 per fully analysed clip, single-digit seconds batched. Qwen adds
4–8 s for both registers, on gated clips only. The detector and Qwen are never
resident together: score → free → load → narrate → free.

`CFXConfig().cpu_fallback()` is the documented CPU profile: no KernelSHAP, IG at
16 steps, one fill, template narration. ~400 forwards/clip.

**Resolution safety.** `FrequencyBranch` and `TSCBranch` precompute a radial ring
index for one input size — feeding them another raises (that was the `index_add_`
crash in an earlier session). Nothing in CFX changes H or W: the scale-jitter TTA
downscales and upscales back, and the spectral counterfactual filters in the
Fourier domain and inverts.

---

## 9. The assertions

These are assertions rather than warnings because a silently miscalibrated run is
the most expensive failure mode in this project — its output looks like a result.

- **`assert_coverage_sane`** raises when the system confidently answers more than
  90% of clips. At ~0.57 held-out AUC most clips *should* abstain; coverage that
  high means a leaky split or a calibrator fitted on test.
- **Numeral fidelity** — every numeral in the technical register appears in the
  card.
- **Shapley efficiency** — a non-zero residual means non-determinism, and is
  reported on the card.
- **Exchangeability** — `check_exchangeability` warns loudly whenever the
  calibration and evaluation manifests differ in provenance. The conformal
  guarantee holds under exchangeability; calibrating on Celeb-DF and evaluating
  on FF++ or DFDC voids it, and no code repairs that.
- **Never one deletion baseline.** Two fills, always, with disagreement flagged.

---

## 10. Status and known gaps

**Tested:** 142 checks in `scripts/test_cfx.py` against a stub model with known
answers — exact Shapley finds the planted decisive branch and gives the noise
branch zero with an exact efficiency residual; IG localises the planted frame and
region; KernelSHAP's top region lands inside the planted patch; a real
attribution beats random on deletion and a random one does not; a hostile
generator writing "clearly a deepfake (99%)" with no citations cannot reach the
user; the abstain path short-circuits at 14 forwards versus 330.

**Untested:** `cfx_fit.py` and `cfx_eval.py` against real checkpoints. They are
thin orchestration over tested functions, but start with `--limit 32` / `--n 8`.

**Gaps worth knowing:**

1. Per-channel AUC in the reliability matrix is an **ablation-delta proxy**, not a
   retrained single-channel model. The JSON says so; don't report it as "channel
   AUC" unqualified.
2. `cfx_fit.py`'s stack features use the ablation deltas rather than per-clip
   descriptor percentiles, which makes it a *lower bound* on arbitration — the
   right direction for a number that has to survive a paired bootstrap.
3. `Config.active_branches()` omits `lmf` (it lists spatial/frequency/srm/tsc
   only). CFX does not use it, so it was left alone rather than changed
   untested — but it is a real bug and anything else calling it is under-counting.
4. Stability is computed per candidate on demand and is not yet aggregated into
   the reliability matrix's `stability` column by `cfx_eval.py`; it reads NaN
   there, which the geometric-mean score treats as zero-ish rather than as
   passing.
5. No real-data comparison against any published deepfake XAI baseline, because
   the datasets are still not obtainable. That gap is unchanged by this work.

---

## 11. One cheap experiment worth doing

Show the plain-language explanation to 10–15 non-technical people, show
verdict-only to a second group, and ask both how often they think the system is
right. If the explanation inflates their estimate, the layman register is doing
harm and you have measured it.

It is the only direct test of whether the layman register achieves what it
claims, it costs an afternoon, and almost nobody in the deepfake XAI literature
runs it.
