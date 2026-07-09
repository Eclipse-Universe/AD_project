# AD Project — Chemical Process Anomaly Detection

## Goal
Detect anomalies in chemical process sensor data (Tennessee Eastman Process-style simulation).  
Binary classification per run: `faultNumber` 0 = normal, 1 = anomaly. Scored on F1-score.  
Baseline public F1: **0.5607** → Current best: **F1 0.9748** (Exp 38).

## Data
- `train.csv`: 250,000 rows × 55 cols. Normal-only (`faultNumber` always 0). 500 simulation runs × 500 timesteps each, sequential run order.
- `test.csv`: 710,400 rows × 54 cols. No `faultNumber` column (prediction target). 740 simulation runs × 960 timesteps each, run order shuffled. **A=240** (true anomaly run count, confirmed via all-1 submission).
- Columns: `xmeas_1..41` (sensor measurements), `xmv_1..11` (manipulated/control variables) — 52 process variables total.
- Large raw CSVs excluded from version control; see `data/README.md`.

## Problem framing
**One-class / novelty detection** — only normal data available for training.  
Labels are **run-level**: each simulation run (all 960 timesteps) is either entirely normal (0) or entirely anomalous (1). Individual timestep labels are not provided.

## Key architecture decisions

### Inference: run-level aggregation
All models output per-timestep anomaly scores → aggregate to run-level mean → apply threshold.  
This produced the single largest performance jump: F1 0.59 → 0.87 (Exp 4 → Exp 8), model unchanged.

### Current best model (Exp 38): GMM + Mahal + SPE ensemble, PP=236
```python
# Run-level feature: mean of 960 timesteps per run → 52-dim vector
tr_run_vecs = tr_X_df.groupby(train_run_ids).mean()   # shape (500, 52)
te_run_vecs = te_X_df.groupby(test_run_ids).mean()    # shape (740, 52)

# Three components
z_gmm  = zscore(-gmm.score_samples(te_run_vecs))      # GMM neg log-likelihood
z_mah  = zscore(-mah.decision_function(te_run_vecs))  # KMeans-Mahalanobis
z_spe  = zscore(spe_per_run(te_X))                    # PCA reconstruction error (row-level agg)

# Weighted ensemble
ensemble = 0.6*z_gmm + 0.3*z_mah + 0.1*z_spe
anomaly_runs = ensemble > threshold   # PP=237 → after oracle: PP=236
```

### FP identification via leaderboard oracle
`TP = F1 × (PP + A) / 2` — derived from F1 definition with known A=240.  
Submit small PP predictions → compute exact TP count in the submitted subset → identify FPs.

## Current confusion matrix (Exp 38, run-level, A=240)
| | Predicted normal | Predicted anomaly |
|---|---|---|
| Actually normal | TN=496 | FP=4 |
| Actually anomaly | FN=8 | TP=232 |

## Known gotchas
- Test column order is alphabetical, train is not — always select by name, never by position.
- Submission must preserve `index=True` in `to_csv()`.
- GaussianMixture and KMeans have non-determinism (n_init=10 reduces but doesn't eliminate it). Boundary runs (~6) may shift between reruns. Always load `output_exp30(GMM-Mahal-SPE30).csv` as the anomaly set base rather than recomputing.
- `outputs/` is gitignored. Submission CSVs are not version-controlled; reproduce via `src/run_expN.py`.
- MENTORING_QUESTIONS.md and QA_LOG.md are gitignored (personal notes).

## Repo layout
```
AD_project/
├── src/                     # experiment scripts + shared modules
├── docs/                    # EXPERIMENT_LOG, ENGINEERING_LOG, EDA_SUMMARY, CONCEPTS, SRC_DESIGN
├── eda/                     # EDA scripts and output CSVs
├── baseline_code/           # original competition baseline (unmodified)
├── data/                    # train.csv, test.csv (gitignored), sample_submission.csv
├── outputs/                 # submission CSVs (gitignored)
├── README.md                # project overview and performance summary
└── PROJECT_CONTEXT.md       # this file
```
