# Data

`train.csv` (91MB) and `test.csv` (253MB) are excluded from this repo via `.gitignore` due to their size.
`sample_submission.csv` is kept since it's small and useful as a format reference.

To reproduce locally, place `train.csv` and `test.csv` in this directory. Expected shapes:

| file | rows | cols | notes |
|---|---|---|---|
| train.csv | 250,000 | 55 | `faultNumber`, `simulationRun`, `sample`, `xmeas_1..41`, `xmv_1..11`; all `faultNumber == 0` (normal only); 500 simulation runs × 500 samples each |
| test.csv | 710,400 | 54 | same sensor columns minus `faultNumber` (alphabetically ordered, unlike train); 740 simulation runs × 960 samples each, normal/abnormal runs shuffled |
| sample_submission.csv | 710,400 | 2 | index + `faultNumber` prediction template (0 = normal, 1 = anomaly) |
