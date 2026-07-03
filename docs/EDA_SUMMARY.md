# EDA 요약 — 전처리 설계 근거

분석일: 2026-07-02 | 피처 수: 52 | 스크립트: eda/preprocessing_eda.py

---

## 1. Scaler 선택: StandardScaler 우선, RobustScaler 비교 실험 예정

52개 피처 전부 |왜도| < 0.2 → 정규분포에 가깝다 → StandardScaler의 이론적 가정(정규성)은 충족.

단, 이론이 실제 성능과 일치하지 않을 수 있다:
- Train 정상 데이터에도 센서 노이즈·일시적 스파이크 존재 → StandardScaler의 평균/분산이 이 값에 영향받을 수 있음
- RobustScaler(중앙값/IQR 기반)는 이런 극값에 강건
- 다른 팀들이 RobustScaler에서 더 나은 성능을 보고한 사례 있음
→ LOF/KMeans/KNN 실험에서 Standard vs Robust를 로컬 score 분포로 비교한 후 결정.
→ IsolationForest는 스케일 무관(수치 검증 완료: Exp 14 vs 14b 동일 F1=0.8870) — IF에는 스케일러 실험 제출 금지.

---

## 2. Train/Test 분포 이동 (KS-test)

52개 피처 전부 p < 0.05. 모든 피처에서 train/test 분포 차이가 통계적으로 유의미.

### 핵심 이상 신호 피처 (KS-stat 상위 + std_ratio 높은 것)

| 피처 | KS-stat | test/train std 비율 | 비고 |
|---|---|---|---|
| xmv_5 | 0.0590 | **18.46x** | 가장 강한 신호 |
| xmeas_16 | 0.0601 | 7.84x | |
| xmeas_7 | 0.0605 | 6.93x | xmeas_13과 상관 0.997 |
| xmeas_13 | 0.0594 | 6.76x | xmeas_7과 상관 0.997 |
| xmv_4 | 0.0604 | 4.74x | |
| xmeas_11 | 0.0594 | 4.85x | |
| xmeas_34 | 0.0616 | 4.62x | |
| xmeas_19 | 0.0737 | 4.35x | xmv_9와 상관 0.986 |
| xmv_9 | 0.0595 | 4.24x | |
| xmeas_31 | 0.0579 | 4.69x | |

---

## 3. 고상관 피처 쌍 및 처리 방침

총 11쌍 (|r| ≥ 0.90):

| 쌍 | r값 | 처리 방침 |
|---|---|---|
| xmeas_12 ↔ xmv_7 | +1.0000 | xmv_7 제거 확정 |
| xmeas_15 ↔ xmv_8 | +1.0000 | xmv_8 제거 확정 |
| xmeas_17 ↔ xmv_11 | -0.9992 | xmv_11 제거 확정 |
| xmeas_1 ↔ xmv_3 | +0.9966 | xmv_3 제거 확정 |
| xmeas_7 ↔ xmeas_13 | +0.9974 | 보류 — 둘 다 KS 상위, 이상에서 관계 깨질 수 있음 |
| xmeas_19 ↔ xmv_9 | +0.9861 | 보류 — xmeas_19 KS 1위 |
| xmeas_18 ↔ xmv_9 | +0.9701 | 보류 |
| xmeas_7 ↔ xmeas_16 | +0.9695 | 보류 — 둘 다 강한 신호 |
| xmeas_13 ↔ xmeas_16 | +0.9598 | 보류 |
| xmeas_18 ↔ xmeas_19 | +0.9496 | 보류 |
| xmeas_10 ↔ xmv_6 | +0.9483 | 보류 |

즉시 제거 확정: xmv_3, xmv_7, xmv_8, xmv_11 (완전/거의 완전 상관)
나머지는 제거 전후 성능 비교 실험으로 결정.

---

## 4. 다음 실험 설계

### 피처 세트
- Full (52개): 현재 그대로
- Reduced (48개): xmv_3, xmv_7, xmv_8, xmv_11 제거

### 모델 순서 (동일 전처리 — StandardScaler + run-level mean)
1. IF + StandardScaler (exp14) — Q7 검증: IF 자체 vs AE 성능 차이 원인 분리
2. LOF (exp15)
3. KMeans 거리 기반 (exp16)
4. KNN 거리 기반 (exp17)
5. 앙상블 (exp18+)
