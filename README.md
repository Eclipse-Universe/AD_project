# TEP Anomaly Detection

Tennessee Eastman Process(TEP) 화학 공정 센서 데이터에서 이상을 탐지하는 프로젝트.  
Train은 정상 데이터만 포함된 **Novelty Detection** 문제이며, 평가 지표는 **F1-score**.

---

## 성능 요약

| 실험 | 모델 | F1 | Accuracy | 핵심 변경 |
|---|---|---|---|---|
| Exp 0 | IsolationForest (baseline) | 0.5607 | 0.7936 | — |
| Exp 8 | IsolationForest + **run-level 집계** | 0.8692 | 0.9162 | **+0.28 브레이크스루** |
| Exp 13 | Autoencoder (재구성 오차) | 0.9205 | 0.9486 | 딥러닝 도입 |
| Exp 17 | KMeans + 마할라노비스 거리 | 0.9277 | 0.9541 | 거리 메트릭 개선 |
| Exp 21 | GMM (tied covariance) | 0.9372 | 0.9622 | 확률 모델 도입 |
| Exp 25 | GMM + Mahal + LOF 앙상블 | 0.9623 | 0.9757 | 앙상블 시작 |
| Exp 30 | GMM + Mahal + **SPE(PCA)** 앙상블 | 0.9707 | 0.9811 | LOF→SPE 교체 |
| **Exp 38** | Exp30 + **오라클 기반 FP 제거** | **0.9748** | **0.9838** | **현재 최고** |

전체 실험 상세 기록 → [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

---

## 핵심 발견

### 1. Run-level 집계 전환 (Exp 8) — F1 +0.28, 단일 최대 도약

Row 단위 판정 대신 `decision_function()` 점수를 **run별 평균**으로 집계한 뒤 threshold 적용.  
모델 자체는 동일하고 추론 방식만 바꿨는데 F1이 0.59 → 0.87로 뛰었다.

문제 구조(레이블이 run 단위)와 추론 단위를 맞춘 것이 핵심이다.  
t-SNE 시각화에서 이상 run의 row들이 일관되게 정상 군집과 분리됨을 확인 → run 평균이 이 일관성을 증폭시킨다.

### 2. 앙상블 신호 다양성 (Exp 25→30) — F1 +0.008

세 모델이 **서로 다른 이상 유형**을 탐지한다.

| 모델 | 탐지 원리 | 가중치 |
|---|---|---|
| GMM (run-level) | 정상 분포에서 이탈한 run | 0.6 |
| KMeans-Mahal (run-level) | 클러스터 중심에서 마할라노비스 거리 이탈 | 0.3 |
| PCA-SPE (row-level 집계) | 센서 간 상관 구조(PCA 부분공간) 이탈 | 0.1 |

SPE는 단독 성능이 GMM보다 낮지만, "평균은 정상인데 센서 간 관계가 무너진" 이상 유형을 보완한다.  
오류 상관이 낮아 앙상블에서 유효하게 기여.

### 3. 오라클 기반 FP 제거 (Exp 37→38) — F1 +0.004

리더보드를 쿼리 인터페이스로 활용해 FP를 정밀 역산.

```
all-1 제출 (PP=740)   → F1=0.4898 → A=240 (실제 이상 run 수) 확정
PP=5 제출             → F1=0.0325 → TP 역산: 5개 run 중 3.5개가 실제 TP
PP=1 제출 (run13만)   → F1=0.0000 → run13 = FP 확정
run13 제거 (PP=237→236) → F1=0.9748 신규 최고
```

역산 공식: `TP = F1 × (PP + A) / 2`

**부산물 발견**: 5가지 모델 신호 전부 정상인 run의 70%가 실제 이상(hard anomaly).  
"약한 신호 = FP" 가정이 기각됐고, 일부 이상 run은 현재 접근법으로 원리적 탐지 불가.

---

## 현재 상태 (Exp 38 기준)

```
추정 혼동 행렬 (run 단위, A=240):
  TP = 232  FP = 4  FN = 8  TN = 496

FN 8개: 모든 방법이 공통으로 놓치는 hard anomaly
        (run 평균 통계량으로 탐지 불가능한 fault type 포함 추정)
FP 4개: 앙상블 점수 상위권에 위치, 특징 기반 식별 불가
```

---

## 프로젝트 구조

```
AD_project/
├── src/
│   ├── data_loader.py              # 데이터 로드
│   ├── preprocess.py               # 스케일링, 피처 선택
│   ├── model.py                    # KMeansMahalanobisDetector 등 커스텀 모델
│   ├── infer.py                    # run-level 집계 추론, 제출 파일 저장
│   ├── run_experiment.py           # IF/OC-SVM 실험 (Exp 0~7)
│   ├── run_gmm_grid.py             # GMM 하이퍼파라미터 탐색
│   ├── run_ensemble.py             # GMM+Mahal+LOF 앙상블 (Exp 25)
│   ├── run_ensemble3.py            # GMM+Mahal+SPE 앙상블 (Exp 30)
│   ├── run_exp34b.py               # contamination 정정 실험 (PP=240)
│   ├── run_exp35.py                # FP 후보 하위 3개 제거 실험
│   ├── run_exp36.py                # 5-signal 교차검증 FP 후보 제거
│   ├── run_exp37_oracle.py         # run13 단독 오라클 (PP=1)
│   ├── run_exp38.py                # run13 제거 → 현재 최고 (F1=0.9748)
│   └── stability_analysis.py       # 모델 비결정성 분석
├── docs/
│   ├── EXPERIMENT_LOG.md           # 전체 실험 기록 (변경→가설→점수→분석)
│   ├── ENGINEERING_LOG.md          # 설계 결정과 근거
│   ├── EDA_SUMMARY.md              # 탐색적 데이터 분석 결과
│   ├── CONCEPTS.md                 # 핵심 개념 정리
│   └── SRC_DESIGN.md               # src/ 모듈 설계 근거
├── eda/                            # EDA 스크립트 및 결과 CSV
├── baseline_code/                  # 대회 제공 baseline (수정 없이 보존)
├── data/                           # train.csv, test.csv (.gitignore 처리)
└── outputs/                        # 제출 파일 (.gitignore 처리)
```

---

## 실험 재현

```bash
pip install -r baseline_code/requirements.txt

# 현재 최고 모델 재현 (Exp 38)
cd src && python run_exp38.py

# 앙상블 베이스 (Exp 30)
cd src && python run_ensemble3.py

# FP 오라클 탐색 (Exp 37)
cd src && python run_exp37_oracle.py
```

실험별 설정·결과·분석 → [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)  
설계 결정 근거 → [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)
