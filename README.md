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
jupyter nbconvert --execute --inplace project.ipynb     # ~25 min
pdflatex report.tex && pdflatex report.tex              # twice, for cross-references
```

The notebook is deterministic: every random seed is derived from
`(model, run, participant)` via `stable_seed()`, the recovery simulation uses
`seed=7`, and all bootstraps use `seed=0`. Re-running reproduces the reported
numbers exactly.

Runtime is dominated by the fitting (about 40 s per run for all four models),
the recovery simulation (200 replicates x 3 models, about 80 s) and the
bootstrap confidence intervals.

## Pipeline

1. **Data** -- loading, cleaning (non-responses, RT < 250 ms), per-participant
   quality diagnostics, model-free indifference analysis, and a run A vs run B
   comparability check (the two runs differ in design and in behaviour, which
   matters for both reliability and out-of-sample prediction).
2. **Models** -- Rachlin hyperboloid valuation + logistic choice rule (held
   fixed), shifted log-normal RTs (held fixed), and three value->RT mappings:
   `M1_abs` (linear conflict), `M2_unc` (decision uncertainty), `M3_sq`
   (quadratic conflict), plus the choice-only `baseline`.
3. **Inference** -- per-participant MAP on run A, L-BFGS-B, 10 restarts,
   weakly informative priors; MLE and likelihood-curvature diagnostics.
4. **Parameter recovery** -- simulate, refit with and without RT, quantify.
5. **Reliability** -- independent fits to run A and run B, test-retest.
6. **Out-of-sample evaluation** -- run A parameters applied to run B, choice and
   RT metrics, reward and loss reported separately.
7. **Does RT help?** -- RT models compared on the RT predictive log-likelihood;
   RT-informed vs choice-only compared on choice quantities.
8. **Recommendation, limitations and next steps.**
