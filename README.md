# Modelling choice and reaction time in probability discounting

Final project, Computational Cognitive Science, Summer Term 2026.
Ashhad Raza Quadri, Saliq Neyaz, Maria Alejandra Pabon Galindo.

## What is here

| File | Contents |
|---|---|
| `project.ipynb` | The complete analysis pipeline, sections 1--8. Runs top to bottom from the raw `.mat` files with no manual intervention. |
| `report.tex` / `report.pdf` | The report (LaTeX source and compiled PDF), using `macros_FP.tex`. |
| `ai_disclosure_*.tex` | One AI usage disclosure per group member. |
| `PD_data/` | Probability discounting data set (49 participants). This is the data set used. |
| `DD_data/` | Delay discounting data set, not used. |
| `figures/` | Figures written by the notebook; the report includes these files directly. |
| `results/` | Result tables written by the notebook as CSV; every number quoted in the report comes from here. |

## Reproducing

```bash
pip install -r requirements.txt
jupyter nbconvert --execute --inplace project.ipynb     # ~16 min
pdflatex report.tex && pdflatex report.tex              # twice, for cross-references
```

The notebook is deterministic: every random seed is derived from
`(model, run, participant)` via `stable_seed()`, the recovery simulation uses
`seed=7`, and all bootstraps use `seed=0`. Re-running reproduces the reported
numbers exactly.

Runtime is dominated by the two recovery simulations (200 replicates x 3 models
x 2 fits, about 4.5 min each for the run A and run B designs), the per-participant
fitting (about 2 min per run for all four models) and the bootstrap CIs.

## Pipeline

1. **Data** -- loading, cleaning (non-responses, RT < 250 ms), per-participant
   quality diagnostics, model-free indifference analysis (run A only), and a run A
   vs run B comparability check. **The two runs are not the same task:** `p_cert`
   is absent in run A (the smaller option is certain, sure-vs-risky) and present in
   run B (0.3-0.7, risky-vs-risky). This governs how sections 5 and 6 are read.
2. **Models** -- Rachlin hyperboloid valuation applied to **both** options,
   `V_i = r_i / (1 + h*theta_i)`, which reduces to `V_cert = r_cert` exactly when
   `theta_cert = 0` (run A); logistic choice rule (held fixed); shifted log-normal
   RTs (held fixed); and three value->RT mappings:
   `M1_abs` (linear conflict), `M2_unc` (decision uncertainty), `M3_sq`
   (quadratic conflict), plus the choice-only `baseline`.
3. **Inference** -- per-participant MAP on run A, L-BFGS-B, 10 restarts,
   weakly informative priors; MLE and likelihood-curvature diagnostics.
4. **Parameter recovery** -- simulate, refit with and without RT, quantify;
   repeated over run B designs (4.4) to ask whether `h` is recoverable in principle
   where `theta_cert > 0`, separately from whether it transfers.
5. **Reliability** -- independent fits to run A and run B. Because the runs are
   different task structures this is generalisation of a construct across decision
   contexts rather than classical test-retest.
6. **Out-of-sample evaluation** -- run A parameters applied to run B, choice and
   RT metrics, reward and loss reported separately; plus a same-task control (6.1)
   that fits half of run A and predicts the other half, separating "the model is
   poor" from "a sure-vs-risky fit does not extrapolate to risky-vs-risky".
7. **Does RT help?** -- RT models compared on the RT predictive log-likelihood;
   RT-informed vs choice-only compared on choice quantities.
8. **Recommendation, limitations and next steps.**
