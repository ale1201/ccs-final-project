# Development Log — Modelling Choice and Reaction Time in a Probability Discounting Task

**Computational Cognitive Science, Final Project**

This document records the full modelling pipeline as it was actually built: the decisions
taken, the reasoning behind them, and — importantly — the points where earlier code had to be
revised once the data revealed something we had assumed wrongly. It is meant to be read
alongside the notebook, which contains the executable version of everything described here.

---

## Overview and goal

The task was to model two behavioural modalities jointly — **binary choice** and **reaction
time (RT)** — in a probability discounting experiment, under the constraint that both are
driven by the same latent quantity: the subjective value difference ΔV between the two options
on each trial. One valuation model and one choice rule are held fixed; several RT models are
specified that differ only in how ΔV maps onto response time. The pipeline covers
specification, inference, identifiability, reliability, out-of-sample prediction, model
comparison, and a reasoned recommendation.

**Dataset chosen: Probability Discounting (PD).** 49 participants, two runs of 200 trials each.
Chosen over the delay-discounting set because the sample is homogeneous, the odds column comes
pre-computed, and it is roughly half the computational effort.

---

## Step 1 — Data loading, inspection, and cleaning

### 1.1 Loading and the label check

The `.mat` files were loaded with `scipy.io.loadmat` (not pandas, despite the brief's wording),
reshaped into one long-format DataFrame with `participant_id` / `run` / `trial` identifier
columns. This long format was kept throughout — it is far easier to group and plot than a dict
of per-participant frames.

**Decision — validate the column layout instead of trusting it.** An early version of the
loader read `data_labels` but never used it. This was flagged and fixed: the label strings are
compared against an `EXPECTED` list on every file, and the loader raises if they diverge. This
check immediately paid off — it confirmed the column order and, crucially, the coding
`action: 1 = certain, 2 = uncertain, 0 = missing`, which anchors the sign of every parameter
downstream (`y = (choice == 2)`).

### 1.2 What inspection revealed (and corrected assumptions)

Several checks were run before any cleaning rule was fixed, and each one changed a decision:

- **Missing choices are trivial and self-consistent.** 68 of 19,600 trials have `choice == 0`,
  and exactly those 68 have `rt == NaN`. So missingness needs no modelling; the trials are
  dropped with one sentence in the report. These are the experiment's timeouts.

- **`prob` is in percent, not proportion.** The column takes values 10/25/50/75/90. Computing
  odds as `(1 - p)/p` on those raw numbers would have produced *negative* odds and corrupted
  every subjective value silently. Fixed by dividing by 100 before computing θ. A verification
  `assert np.allclose(theta, odds)` confirms the derived θ matches the provided odds column.

- **RTs are in milliseconds** (median ≈ 1938 ms). Divided by 1000 once, at cleaning time.

- **Condition is within-participant and within-run.** The crosstab shows 100 trials in every
  participant × run × condition cell. This licenses estimating **separate discount rates for
  the reward and loss conditions**, and — at the time — appeared to make run A vs run B a clean
  test–retest comparison. (This last assumption was later overturned; see the Run B correction
  below.)

- **The RT window (200–10000 ms) was checked, not assumed.** An early note claimed the data
  were "hard-bounded" at those values. Direct inspection showed only 2 trials at exactly 200 ms
  and 4 above 9900 ms, with continuous values throughout — i.e. no software floor and **no
  ceiling censoring**. This is why *no upper RT cut-off is applied*: a shifted log-normal can
  model a long tail, but the data show there is no truncation artefact to remove. The claim in
  the markdown was corrected to reflect this.

### 1.3 Cleaning rules and their justification

Order of operations matters and was fixed deliberately:

1. Drop `choice == 0` **first** (their RTs are meaningless).
2. Convert ms → s.
3. Apply the lower RT cut-off.

Filtering on RT before removing non-responses would let timeout-coded RTs decide which trials
survive.

**Lower cut-off at 250 ms.** This is a *theoretical* choice, not an empirical one — the RT
histogram shows no separated cluster of anticipations, so any cut-off is a modelling decision.
Below ~250 ms there is no time to perceive two options, compare them, and execute a response.
It costs 105 trials (0.54%). The model also needs RT > τ with τ meaning genuine non-decision
time, which a 210 ms trial would violate.

**No upper cut-off**, per the censoring check above.

Final retained sample: **19,427 of 19,600 trials (99.1%), 49 participants**.

### 1.4 Per-participant quality control

A `QC` table (one row per participant) was built to (a) justify the estimator later and (b)
flag participants for the RT analysis. Key findings:

- **Zero participants with extreme choice** (>95% one option). This later forced a change in how
  the priors were justified — the usual "MLE diverges for always-one-option participants"
  argument does not apply here.
- **Four participants respond very fast** (>10% of trials below 400 ms). `bh348lli7` is extreme:
  57% of trials under 400 ms, median RT 379 ms vs 1950 ms for the sample. These matter for the
  RT analysis specifically, and were tracked (not auto-excluded) so that any decision could be
  made on evidence in the evaluation step.

### 1.5 Model-free h as a validation anchor

Before fitting anything, an independent estimate of the discount rate was derived from the
indifference points. At indifference V_unc = V_cert, which rearranges to
**ratio = 1 + h·θ** — a line with slope h. Fitting a simple logistic per (condition ×
probability) cell and reading off the 50% point gave indifference ratios that fall close to
that line.

**This was not required by the brief, but was kept deliberately** because it is the only way to
catch a sign or scale error in the later likelihood: the optimiser always returns *some* number,
and without an independent reference there is no way to know if it is right. This anchor caught
a real problem later (see the Run B correction).

A bug here was found and fixed: an early version filtered indifference fits by `slope > 0`,
which silently discarded the entire **loss** condition, where the logistic slope is negative by
construction. Fixed by requiring the sign to match the condition. Once corrected, both
conditions produced clean lines, and the **sign effect** was clear.

---

## Step 2 — Model specification

### 2.1 Valuation (fixed) — hyperbolic discounting in odds space

V = r / (1 + h·θ), with θ = (1−p)/p the odds against. Justified from the data, not just
convention: the indifference ratios in Step 1 are linear in θ, which is exactly what the
hyperbolic form predicts and what an exponential form would not. h = 1 corresponds to
risk-neutral (expected-value) behaviour, giving a principled reference point.

**Two rates per participant (h_gain, h_loss),** justified by the near order-of-magnitude
difference between the two model-free rates.

**Sign handling — a corrected assumption.** The initial plan assumed positive magnitudes with a
manual sign flip for losses. Inspection showed the magnitudes are **already stored signed**
(gains positive, losses negative, verified by assertion). So *no* sign correction is applied:
the same equation then produces both risk attitudes automatically (risk aversion for gains,
risk seeking for losses — the reflection effect), with no extra parameter. Applying the
originally-planned manual flip would have inverted this. The `sgn` column from the first draft
was removed.

**Scaling.** ΔV is divided by (|r_cert| + |r_unc|)/2. This is purely numerical — magnitudes span
five orders of magnitude, so without normalisation no single β could serve both small and large
trials. Documented as a modelling convenience, not a psychological claim.

### 2.2 Choice rule (fixed)

P(uncertain) = σ(β·ΔV + β₀).

### 2.3 RT distribution (fixed) — shifted log-normal

RT = τ + LogNormal(μ, σ). Held identical across all RT models so the comparison isolates the
value→RT mapping (a distributional change would not count as a distinct model). Chosen because
the RT histogram is right-skewed, RTs are positive, and τ has a clear meaning (non-decision
time). τ is parameterised through a sigmoid onto (0, min RT) so it can never exceed the fastest
response.

### 2.4 The RT models

Three mappings, all with the same eight-parameter footprint, differing only in μ:

| Model | μ | Hypothesis | Marginal effect as \|ΔV\| grows |
|---|---|---|---|
| **M1** linear conflict | a − b·\|ΔV\| | Value distance slows the comparison | Constant |
| **M2** decision uncertainty | a − b·\|2P−1\| | Subjective indecision slows it; saturates once the choice is clear | Shrinks |
| **M3** quadratic conflict | a − b·ΔV² | Only near-ties are costly | Grows |

**Sequencing of the third model.** The pipeline was first built and validated end-to-end with
**two** RT models (M1, M2), and M3 was added afterwards once the machinery was known to work.

**A scale correction for M3.** Because ΔV² is on a smaller scale than |ΔV| (median ≈ 0.07 vs
0.27 in run A), an equivalent effect on μ requires a larger b. Using one prior/one generating
range for all models would have penalised M3's regularisation and made its recovery look
artificially poor. Fixed with model-specific `PRIOR_B` (wider for M3) and a model-specific
generating range `GEN_B` in the recovery step.

---

## Step 3 — Inference on run A (MAP, per participant)

### Estimator and its justification

**Per-participant MAP**, run A only. Per-participant rather than hierarchical was chosen
because hierarchical shrinkage would inflate the reliability correlation (Step 5) and blur the
recovery question (Step 4) — both required analyses depend on independent, unshrunk estimates.

Parameters fitted on unconstrained scales (log h, log β, log σ, τ_raw) with weakly informative
Gaussian priors. Optimisation via L-BFGS-B with **10 random restarts** (first from the Step 1
estimates, the rest from the priors), because these likelihoods have local optima — a single
start is not enough.

A reproducibility bug was fixed here: Python's built-in `hash()` is randomised per process for
strings, so seeds derived from it were not reproducible across sessions. Replaced with an
md5-based `stable_seed`.

### Justifying regularisation with evidence, not assertion

Since no participant chose one option almost always, the usual justification for priors did not
apply. A **curvature diagnostic** was built instead: how much log-likelihood is lost when each
parameter is displaced by 0.5 on the fitted scale. This identified **19 of 49 participants**
with at least one flat direction, concentrated in the discount rates (14 flat in log_h_gain,
14 in log_h_loss, but only 3 in log_beta). The narrow range of magnitude ratios in the loss
condition explains why h is the fragile parameter.

For the 30 well-identified participants, **MAP and MLE coincide almost exactly** (r = 0.995–0.999,
median deviation < 0.13), confirming the priors regularise only the ill-conditioned cases
without biasing the rest. An initial version of this comparison used a weak "at the bound"
detector that misclassified several unidentified participants as identified and depressed the
correlation to ~0.74; switching to the curvature criterion fixed it and told the correct story.

Result: 100% convergence for baseline/M1/M3, 94% for M2.

---

## Step 4 — Identifiability / parameter recovery

Synthetic data were generated from known parameters on real trial designs, then re-fit with the
identical pipeline. Crucially, **each synthetic dataset was fit twice** — with RT and
choice-only — so the "does RT sharpen recovery?" comparison is paired.

**Main findings:**

- Recovery is good for 7 of 8 parameters (r = 0.82–0.98).
- **τ is not identifiable** (r ≈ 0.2, the one clear exception among the 8 recovered parameters).
  The recovery matrix (§4.3) shows why: τ's recovered value correlates with the *true* value of
  other parameters (notably `a` and `b`) almost as much as with its own true value, indicating a
  trade-off rather than a clean one-to-one mapping — invisible in the final estimates and only
  visible against the known ground truth, a direct illustration of why this section is required.
  τ is a nuisance parameter, so this does not threaten any conclusion.
- **RT improves recovery of the discount rates**, most strongly under M3 (RMSE ratio 0.47 for
  log_h_gain vs 0.86 for M1, 0.98 for M2).

**Comparing recovery to the real effect size (Step 7).** The real slope of log RT on |ΔV|
(gains, using M1's fitted h) is only −0.027 within-participant / +0.063 raw pooled — far
smaller than the effect sizes `GEN_B` draws for recovery, and small enough that even the sign
is sensitive to how participant-level RT offsets are handled. This is the likely reconciliation
between the simulation (RT sharpens recovery, most strongly under M3) and the real data (RT does
not detectably improve choice prediction or reliability): recovery was tested across a
*realistic range* of difficulty effects, but real participants sit at the weak end of that
range.

### 4.4 Recovery under run B's task structure (added after the Run B correction)

Once run B was recognised as a structurally different task (below), a second recovery study was
added: is h even *recoverable in principle* from a design with θ_cert > 0, setting cross-run
transfer aside? Answer: it is recoverable but **less well** (e.g. M1 log_h_gain r drops from
0.93 to 0.89, RMSE rises from 0.57 to 0.80). This isolates one contribution to the low
reliability from the transfer question entirely.

---

## The Run B correction — the most consequential rework

Late in development, extending the valuation to handle a `p_cert` column revealed that the two
runs are **not the same task**:

- **Run A:** `p_cert` is NaN for every trial → the smaller option is *certain* (θ_cert = 0).
  Classic sure-vs-risky discounting.
- **Run B:** `p_cert` is present for every trial (values 0.3–0.7) → **both** options are
  probabilistic. The larger-magnitude option spans a wide probability range (0.1–0.9).

The column labels remain `r_cert`/`r_unc` in run B for consistency with run A, even though
neither option is certain there — what is structurally consistent across both runs is
magnitude, so run B is really **small-vs-large, both risky**.

**What was wrong.** The original `compute_dv` treated `r_cert` as undiscounted — correct for run
A (θ_cert = 0) but **wrong for all 9,700 run B trials**, where the "safe" option had probability
0.3–0.7 and was being valued as if certain. This was a genuine modelling error, not a stylistic
choice.

**The fix.** Discount both options with the same equation:
`V_cert = r_cert / (1 + h·θ_cert)`, which reduces correctly to `V_cert = r_cert` when
θ_cert = 0. Steps affected:

- **Step 3 (run A): unchanged** — θ_cert ≡ 0 there, so the new formula is mathematically
  identical. Verified: the run A medians did not move.
- **`H_EMP` recomputed on run A only**, since the ratio = 1 + h·θ derivation assumes an
  undiscounted safe option. This resolved a discrepancy we had previously waved away: the
  model-free h_loss (0.35) and the fitted h_loss (0.82) had differed by 2.3×; after restricting
  to run A the model-free value rose to 0.64 and the two agreed. **The discrepancy had been the
  specification error all along** — a clean confirmation the fix was correct.
- **Steps 5–7 re-run** with the corrected ΔV, plus `theta_cert` added to the recovery design
  columns.

**Why we did NOT revert to the pre-correction version, even though its numbers looked better.**
The uncorrected version gave a reliability of r = 0.44 for h_gain; the corrected version gives
r = 0.18. The temptation to keep the higher number was explicitly rejected: the 0.44 was an
artefact of both runs sharing the same valuation error, not evidence of a stable trait.
Reintroducing a known modelling error to inflate a result would be an integrity problem, not a
modelling choice. The corrected, lower number is the honest one.

**Reframing that follows from the correction.** Because A and B are different tasks under a
single assumed h, Step 5 is no longer test–retest reliability in the strict sense but
**generalisation of the construct across decision contexts**, and Step 6 is **extrapolation to
an unseen decision structure**, not held-out prediction of the same task. This is stated
explicitly in the report.

---

## Step 5 — Reliability / cross-context generalisation

Each model fit independently on runs A and B; parameters correlated across participants
(Pearson, Spearman, ICC, with bootstrap CIs).

**Findings after the correction:**

- h_gain reliability is **low** (r = 0.18, CI includes zero). The discount rate barely
  generalises between the two tasks.
- Differences between RT models are not distinguishable from noise at n = 49 — most CIs cross
  zero.
- `log_sigma` reliability was found to be inflated by Pearson (0.67 vs Spearman 0.46), driven by
  a few participants with extreme RTs; Spearman is reported for it.
- The stable quantities are the RT *level* `a` (r ≈ 0.75) and, more weakly, choice sensitivity;
  the RT difficulty effect `b` and the loss discount rate are not stable.

This is a legitimate, if less flattering, result: h is more context-dependent than a single-trait
model assumes.

---

## Step 6 — Out-of-sample evaluation (run B)

Run A parameters applied unchanged to run B. Metrics for **both modalities**, reward and loss
reported separately. Comparisons follow the brief's warning strictly: RT models are compared on
the **RT predictive log-likelihood** (their shared choice term cancels), and RT-informed vs
choice-only is compared on **choice quantities**.

**Findings after the correction:**

- Choice prediction across tasks is weak in absolute terms (AUC ≈ 0.59, close to chance). An
  earlier draft cited AUC ≈ 0.75 as evidence of good generalisation; that number came from the
  uncorrected ΔV and was removed. The model does **not** extrapolate well between tasks.
- **M2 fits RT best out-of-sample** (p = 0.053 vs M1, p = 0.006 vs M3), and this time the
  in-sample advantage generalises.

### The same-task control — the decisive diagnostic (added late)

To prove the weak cross-task prediction is about the tasks, not a broken model, a control was
added: fit on one half of run A, predict the other half.

- **Same task:** −0.464 LL/trial, **49/49 participants beat chance.**
- **Cross task (A → B):** −1.004 LL/trial, **6/49 beat chance.**

This demonstrates conclusively that the model predicts well *within* a task and fails *between*
different tasks — the low reliability is a property of the data (two distinct tasks), not a
modelling failure. Without this control, r = 0.18 would look like a failure; with it, it is a
demonstrated finding.

---

## Step 7 — Does RT help?

Using the required choice-only baseline (the identical model with the RT term deleted):

- **(a) Choice prediction:** no reliable improvement. Δ choice LL = −0.037 (M1), +0.002 (M2),
  +0.025 (M3), none significant. Modelling RT does not sharpen out-of-sample choice prediction.
- **(b) Recovery/reliability:** in simulation RT sharpens discount-rate recovery (strongly for
  M3); in the real data it does not improve reliability distinguishably. The real difficulty
  effect (slope −0.027 within-participant / +0.063 raw pooled) is far weaker than the range
  recovery was tested across, which is the likely reconciliation between the two results.

---

## Summary of the reworks and why they were made

| Rework | Trigger | Why it mattered |
|---|---|---|
| Validate labels, not trust them | loader ignored `data_labels` | caught column order and choice coding |
| `prob` ÷ 100 | odds mismatch check failed | negative odds would corrupt all values |
| Remove manual sign flip | magnitudes found already signed | flip would invert the reflection effect |
| Fix indifference-fit sign filter | loss condition silently dropped | recovered the sign effect |
| Curvature diagnostic for priors | no extreme-choice participants | evidence-based justification of regularisation |
| `stable_seed` (md5) | `hash()` non-reproducible | reproducibility across sessions |
| Model-specific prior/range for M3 | ΔV² on a smaller scale | fair model comparison |
| Compare recovery's range to the real effect size | M3's wider range could flatter it | reconciles simulation vs real data |
| **Discount both options (Run B)** | `p_cert` present only in run B | corrected a real valuation error on half the trials |
| `H_EMP` from run A only | ratio = 1 + h·θ assumes θ_cert = 0 | resolved the h_loss discrepancy |
| Same-task control | reliability collapsed after fix | proved the collapse is task-difference, not model failure |
| Remove AUC-0.75 claim | came from uncorrected ΔV | honest reporting |

---

## What the corrected analysis concludes

- The hyperbolic-in-odds valuation with a logistic choice rule fits the choices well **within a
  task** (same-task control: 49/49 beat chance).
- The discount rate is **context-dependent**: it does not generalise from a sure-vs-risky task
  to a risky-vs-risky task (reliability r ≈ 0.18; cross-task 6/49 beat chance).
- Among the RT mappings, **M2 (decision uncertainty) describes response times best** out of
  sample, and this replicates from in-sample.
- **Modelling RT was not worthwhile for the choices**: no reliable gain in choice prediction,
  recovery, or reliability, because the empirical difficulty effect (slope −0.027
  within-participant / +0.063 raw pooled) is far weaker than the range recovery was tested
  across.
- The most robust methodological lessons come from the controls that were added along the way:
  the τ–a trade-off (recovery), the real-vs-simulated effect-size comparison (why simulation and
  data disagree), and the same-task control (why cross-task reliability is low).
