# TEP Anomaly Detection

Tennessee Eastman Process(TEP) 화학 공정 센서 데이터에서 이상을 탐지하는 프로젝트.
Train은 정상 데이터만 포함된 **Novelty Detection** 문제이며, 평가 지표는 **F1-score**.

---

## 성능 요약

| 실험 | 모델 | F1 | Accuracy | 핵심 변경 |
|---|---|---|---|---|
| Exp 0 | IsolationForest (baseline) | 0.5607 | 0.7936 | — |
| Exp 8 | IsolationForest + **run-level 집계** | 0.8692 | 0.9162 | F1 +0.28 브레이크스루 |
| Exp 13 | Autoencoder (재구성 오차) | 0.9205 | 0.9486 | 딥러닝 도입 |
| Exp 15 | LOF + StandardScaler | 0.9237 | 0.9514 | ML로 AE 초과 |
| **Exp 17** | **KMeans + 마할라노비스 거리** | **0.9277** | **0.9541** | **현재 최고** |

---

## 핵심 발견

**1. Run-level 집계 (Exp 8)** — 가장 큰 단일 개선 (+0.28 F1)

Row 단위 판정 대신 `decision_function()` 점수를 run별로 평균 → run 전체를 정상/이상 판정.
문제 구조(레이블이 run 단위)에 맞춘 추론 방식 변경만으로 F1이 0.59 → 0.87로 도약.

**2. 마할라노비스 거리 (Exp 17)** — 유클리드 거리의 구조적 한계 극복

유클리드 거리는 `xmv_7 ↔ xmeas_12 (r=1.0)` 같은 피처 간 상관 구조를 무시한다.
이상 run은 개별 센서값이 정상 범위 내에 있어도 센서 간 관계가 깨지는 "형태 B 이상"으로 나타날 수 있다.

마할라노비스 거리 `d(x, μ_k) = √[(x−μ_k)ᵀ Σ⁻¹ (x−μ_k)]` 는 공분산 역행렬 Σ⁻¹에 피처 간 상관 구조가 인코딩되어 있어 이 신호를 포착한다.

- Separation index: 83 → **461** (5.5배 향상)
- FN 행 수: 24,060 → **14,284** (−41%)
- F1: 0.9083 → **0.9277**

---

## 프로젝트 구조

```
AD_project/
├── src/                          # 실험 파이프라인 (모듈화)
│   ├── data_loader.py            # 데이터 로드
│   ├── preprocess.py             # 스케일링, 피처 선택
│   ├── model.py                  # IF, LOF, KMeans, KMeans-Mahal, AE
│   ├── infer.py                  # run-level 집계 추론
│   ├── run_experiment.py         # IF/AE 실험 실행
│   ├── run_lof_grid.py           # LOF 그리드 탐색
│   ├── run_kmeans_grid.py        # KMeans-Euclidean 그리드 탐색
│   └── run_kmeans_mahal_grid.py  # KMeans-Mahalanobis 그리드 탐색
├── docs/
│   ├── EXPERIMENT_LOG.md         # 전체 실험 기록 (변경→가설→점수→분석)
│   ├── ENGINEERING_LOG.md        # 설계 결정과 근거
│   ├── EDA_SUMMARY.md            # 탐색적 데이터 분석 결과
│   ├── CONCEPTS.md               # 핵심 개념 정리
│   └── SRC_DESIGN.md             # src/ 모듈 설계 근거
├── eda/                          # EDA 스크립트 및 결과 CSV
├── baseline_code/                # 대회 제공 baseline (수정 없이 보존)
├── data/                         # train.csv, test.csv (.gitignore 처리)
└── outputs/                      # 제출 파일 output_expN(.csv) (.gitignore 처리)
```

---

## 실험 재현

```bash
pip install -r baseline_code/requirements.txt

# 그리드 탐색 (LOF / KMeans-Euclid / KMeans-Mahal)
cd src && python run_lof_grid.py
cd src && python run_kmeans_grid.py
cd src && python run_kmeans_mahal_grid.py
```

실험별 설정·결과·분석 → [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)
설계 결정 근거 → [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)
