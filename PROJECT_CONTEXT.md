# AD Project — Chemical Process Anomaly Detection

## Goal
Detect anomalies in chemical process sensor data (Tennessee Eastman Process-style simulation).
Binary classification per row: `faultNumber` 0 = normal, 1 = anomaly. Scored on F1-score.
Baseline public F1: **0.5607**.

## Data
- `train.csv`: 250,000 rows × 55 cols. Normal-only (`faultNumber` always 0). 500 simulation runs × 500 samples each, sequential run order.
- `test.csv`: 710,400 rows × 54 cols. No `faultNumber` column (prediction target). Columns are **alphabetically ordered**, unlike train. 740 simulation runs × 960 samples each, run order shuffled (normal/abnormal runs randomly interleaved).
- `sample_submission.csv`: submission format, index + `faultNumber` (0/1).
- Columns: `xmeas_1..41` (sensor measurements), `xmv_1..11` (manipulated/control variables) — 52 process variables total.
- Large raw CSVs (`train.csv`, `test.csv`) are excluded from version control; see `data/README.md`.

## Problem framing
This is **one-class / novelty detection**, not standard supervised classification — only normal data is available for training. The model must learn the boundary of "normal" behavior and flag deviations.

## Baseline approach (`baseline_code/baseline.ipynb`)
1. EDA: distribution histograms, correlation matrix, top correlated/anti-correlated variable pairs visualized over time for `simulationRun == 1`.
2. Feature selection: drop `simulationRun`/`sample`, keep the 52 numeric `xmeas_*`/`xmv_*` columns (no normalization or feature engineering).
3. Models: `IsolationForest` and `SGDOneClassSVM` (scikit-learn), fit on normal-only train data, no validation split.
4. Inference: predict on test, convert sklearn's {1=normal, -1=anomaly} output to {0=normal, 1=anomaly}, save `output.csv` with `index=True`.

## Known gotchas
- sklearn one-class models output `1`/`-1`, not `0`/`1` — must remap before submission.
- `test.csv` columns are alphabetically sorted (different order from `train.csv`) — always select by column name, never by position.
- Submission must preserve `index=True` in `to_csv()`.
- Each row in train/test is treated independently in the baseline; the runs are actually time series (500 or 960 samples per run) — a clear improvement direction is exploiting temporal structure (windows, lag features, per-run aggregation) instead of row-independent modeling.
- Correlation between sensor pairs is computed on normal data only; a broken correlation pattern between two normally-correlated variables may itself be a useful anomaly signal — don't blindly drop "redundant" correlated features.

## Repo layout
```
AD_project/
├── baseline_code/
│   ├── baseline.ipynb
│   └── requirements.txt
├── data/
│   ├── README.md          # data file size/shape notes (large CSVs gitignored)
│   └── sample_submission.csv
├── PROJECT_CONTEXT.md      # this file
└── .gitignore
```
