# Engineering Log — TEP 이상 탐지

설계·전처리·모델 선택에서 내린 **결정과 근거**를 기록한다.
실험별 제출 결과는 EXPERIMENT_LOG.md, EDA 수치는 EDA_SUMMARY.md 참고.

매 세션 시작 시 재독 필수.

---

## 0. 문제 정의 및 구조적 제약

### TEP 데이터 구조
- Train: 500개 simulationRun, 각 500 timestep = 250,000행. **전부 정상(레이블 없음).**
- Test: 740개 simulationRun, 각 960 timestep = 710,400행. 정상+이상 혼재.
- 레이블 단위: **run 전체가 하나의 정상(0) 또는 이상(1)**. 개별 timestep이 아님.
- 피처: xmeas_1~41 (측정값 41개) + xmv_1~11 (조작 변수 11개) = 총 52개.

### 검증 환경 제약
- Train에 정상 데이터만 있어 hold-out F1 계산 불가.
- 로컬 점수(run별 score 분포, bimodality)는 분리 품질의 간접 지표에 불과.
- **리더보드 제출만이 실제 F1을 알 수 있는 유일한 방법.**
- 따라서 제출 횟수는 매우 귀중한 자원. 로컬 실험을 최대한 활용하고, 확신이 없으면 제출하지 않는다.

---

## 1. 추론 설계: Run-level 집계 (핵심 결정)

### 결정 내용
각 row의 decision_function 점수를 run별 평균(mean)으로 집계하고,
집계된 run 점수에 threshold를 적용해 run 단위로 정상/이상 판정한다.

### 결정 근거
- 레이블이 run 단위이므로 row를 독립적으로 판정하면 구조적 불일치 발생.
- t-SNE 시각화에서 이상 run의 row들이 일관된 군집을 형성 — 같은 run 내
  row 점수가 일관되게 높으므로(이상) 평균으로 안정화하면 분리가 선명해진다.
- **Exp 4(row-level IF): F1 0.5869 → Exp 8(run-level IF): F1 0.8692, 단 추론 방식만 교체.**

### 주의사항
- Row 제거나 순서 변경은 일절 없음. 시계열 구조(run 내 timestep 순서) 완전 보존.
- 집계 함수: Exp 10에서 percentile(10th) 시도 → Recall만 급락(0.83), mean으로 복귀 확정.

---

## 2. 피처 선택 전략

### 기본 피처 세트 (52개)
xmeas_1~41 + xmv_1~11. simulationRun, sample 제외(ID 컬럼이므로 피처 아님).

### 행(row) 제거 방침: 절대 금지
시계열 데이터에서 특정 timestep row를 제거하면:
- Run 내 시간 간격이 불균등해짐 (원래 등간격)
- 집계(run-level mean)의 분모가 run마다 달라져 직접 비교 불가
- Diff feature 계산 시 연속하지 않는 두 timestep이 이어져 의미 없는 변화량 생성
따라서 어떤 이유로도 row를 제거하지 않는다.
(→ NaN이 생기는 경우: diff의 첫 timestep은 0으로 채움. Exp 3 참고.)

### 열(column) 제거: 신중하게

**Exp 3 교훈: 피처 추가도 신중하게**
diff feature 52개 추가 (52→104개): F1 0.5812 → 0.5611 악화.
원인: IF는 트리당 랜덤 피처 선택 → 피처 수가 늘면 유용한 피처 선택 확률이 줄어듦.
단일 timestep diff는 정상 운전 중 노이즈가 크고, TEP 고장이 step-change 위주일 경우
추가 정보가 거의 없음. **"피처를 추가하면 항상 좋다"는 가정은 틀렸음을 실험으로 확인.**

**EDA 기반 중복 피처 제거 대상 (4개)**
```
xmv_7  (xmeas_12와 r=+1.0000, 완전 상관)
xmv_8  (xmeas_15와 r=+1.0000, 완전 상관)
xmv_11 (xmeas_17와 r=-0.9992, 거의 완전 상관)
xmv_3  (xmeas_1과  r=+0.9966, 거의 완전 상관)
```

**왜 제거가 필요한가 (거리 기반 모델에서)**
LOF, KNN, KMeans는 Euclidean 거리를 쓴다.
r=1.0인 두 피처(A, B)는 52차원 공간에서 동일한 방향을 가리킨다.
→ 그 방향의 거리 기여가 2배로 계산됨(차원 중복 가중치).
→ A와 B 중 하나를 제거하면 차원 가중치가 균등해져 LOF의 이웃 구조가 더 정확해짐.

**IsolationForest에서는 영향 없음**
IF는 랜덤 분기(수직/수평)만 쓰며 거리를 계산하지 않아 중복 피처의 기하학적 왜곡이 없음.
따라서 IF 실험에는 피처 제거 없이 52개 사용 — 로컬 결과 차이 없을 것이므로 제출 낭비 방지.

**제거 미확정 (보류) 쌍 7개**
- xmeas_7 ↔ xmeas_13 (r=0.997): 둘 다 KS 상위권 — 이상 상황에서 관계가 깨지는 신호일 수 있어 보류
- xmeas_19 ↔ xmv_9 (r=0.986): xmeas_19가 KS 1위 이상 신호 — 제거 후 정보 손실 우려
- 나머지 5쌍: |r| < 0.97, 제거 필요성 낮음

**결정 방법**: 48피처 vs 52피처를 로컬 score 분포(bimodality)로 비교하고 더 명확한 분리를 보인 쪽을 제출.

**LOF 실험 결과 (2026-07-03)**
4가지 조합(Standard/Robust × 52/48피처) 비교:

| 조합 | separation_idx | train std |
|---|---|---|
| Standard + 52피처 (LOF-A) | **68.570** | 0.0164 |
| Robust   + 52피처 (LOF-B) | 68.302 | 0.0165 |
| Standard + 48피처 (LOF-C) | 65.543 | 0.0175 |
| Robust   + 48피처 (LOF-D) | 64.970 | 0.0178 |

**핵심 발견: 피처 제거(52→48)가 LOF에서 역효과**
EDA에서 정상 조건 기준 완전 상관(r≈1.0)이라 제거 "확정"한 4개(xmv_7, xmv_8, xmv_11, xmv_3)를 실제로 제거하자 separation이 내려갔다.
이유: 정상 운전 중 r=1.0인 두 피처가 이상 운전 중에는 이 상관이 "깨짐" → 상관 파괴 자체가 이상 탐지 신호.
xmv_7을 제거하면 xmeas_12-xmv_7 사이의 관계 파괴 신호를 잃는다.
→ "정상 조건에서 중복"과 "이상 탐지에서 중복"은 다른 개념. **거리 기반 모델에서도 피처 제거는 신호 손실 가능성을 고려해야 한다.**

최종 선택: LOF-A (StandardScaler, 52피처) → separation 최고, train std 최소(정상 군집 가장 촘촘)

---

## 3. 스케일러 선택

### 현재 결론 (미확정, 실험 예정)
EDA 기반 이론적 근거: StandardScaler 우선.
실제 성능 비교: LOF 실험에서 Standard vs Robust 비교 예정.

### StandardScaler 근거 (이론)
- 52개 피처 전부 |왜도| < 0.2 — 정규분포에 매우 가까움.
- StandardScaler는 정규분포 데이터에서 평균=0, 분산=1 정규화를 수행하며
  이 조건에서 LOF의 거리 계산이 피처 간 균등하게 이루어짐.

### RobustScaler 가능성 (반론)
- Train 데이터가 모두 "정상"이라도 센서 노이즈, 일시적 스파이크가 존재.
- StandardScaler는 이 극값에 평균/분산이 영향받아 스케일이 왜곡될 수 있음.
- RobustScaler는 중앙값/IQR 기반으로 이 극값에 강건.
- 다른 팀들이 RobustScaler에서 더 좋은 성능을 보고한 사례 존재.
- → 이론만으로 확정하지 말고 반드시 실험으로 비교.

### IsolationForest에서의 교훈 (Exp 14/14b)
IF + StandardScaler (Exp 14) = IF + RobustScaler (Exp 14b) = F1 0.8870.
완전히 동일. 이유: IF는 트리 분기 시 값의 상대적 순위만 사용하며 절댓값 스케일은 무관.
스케일러 변경이 IF 결과에 영향을 줄 것이라는 가정이 틀렸음 — **제출 2회 낭비.**
**앞으로 IF에는 스케일러 실험을 절대 제출하지 않는다.**

---

## 4. 모델 선택 전략

### 실험 순서 (합의됨)
1. IF (row-level) → run-level 집계 브레이크스루(Exp 8) → IF run-level 완성(Exp 12, F1=0.8870)
2. Autoencoder → Exp 13, F1=0.9205 (현재 최고)
3. ML 모델 탐색 (현재 단계): LOF → KMeans → KNN → OC-SVM
4. 앙상블: ML 모델 전부 실험 후

### 왜 ML 모델 탐색인가 (AE 이후에도)
- AE가 0.9205를 줬지만, 무엇이 개선 요인인지 아직 모름.
- ML 모델들은 해석 가능하고 빠름. 각 모델의 특성이 다르므로 앙상블에서 다양성을 줌.
- AE를 바로 개선하기보다 ML 모델 군으로 베이스라인을 넓히는 게 앙상블 전략에 유리.
- DL 실험(AE 개선, 멘토 제안 bottleneck=16)은 ML 탐색 이후로 보류.

### 왜 SGDOneClassSVM이 실패했나 (Exp 5, 7)
- SGDOneClassSVM은 선형 커널만 지원 (SGD 기반이라 속도를 위해).
- TEP 52개 센서 간 관계는 비선형(화학 반응식으로 묶임).
- 선형 경계로는 정상 공간의 복잡한 형상을 표현 못함.
- 비선형 커널 OC-SVM(sklearn.svm.OneClassSVM)은 이론상 맞지만 71만 행에서 수십 분 소요.
- 결론: OC-SVM은 실용적으로 포기. LOF, KMeans, KNN이 대안.

### LOF의 이론적 적합성
- LOF: 로컬 밀도 기반. 각 점의 이웃 밀도를 주변과 비교해 이상 점수 부여.
- TEP 정상 데이터: 화학 반응이 균형점(setpoint)에서 안정적으로 운전 → 정상 데이터가
  고밀도 군집을 이룸. 이상 데이터는 이 군집에서 벗어남 → LOF에 적합한 구조.
- 단, row 수준 거리 계산: 250,000 train × 710,400 test → 무거움.
  novelty=True 옵션 필수 (test에 decision_function 적용 가능하게).

---

## 5. 검증 방법론

### 리더보드 역산
F1 + Accuracy + PP(예측 양성 수, output.csv row count)로 혼동행렬 4칸 역산 가능.
수식: FP+FN = N(1-acc), TP = N×F1×acc / (2-F1) ... (QA_LOG.md 참고)

하지만 더 정확한 방법: **전부 1 or 전부 0 제출**
- 전부 1: Precision = 실제 anomaly 비율 (710,400행 중 실제 이상 행 수)
- 전부 0: Accuracy = 실제 normal 비율
이미 추정 anomaly 비율 ≈ 32.2%가 여러 독립 실험에서 일치해 신뢰도 높음.

### 로컬 score 분포 해석
- run-level score 히스토그램에서 두 봉우리(bimodal) = 정상/이상 분리 양호.
- Train run scores (모두 정상): 좁은 단봉 군집 → 이것이 "정상 기준선".
- Test run scores에서 이 기준선에서 멀리 떨어진 꼬리 = 이상 run.
- 두 봉우리가 선명할수록 threshold 선택이 쉬워지고 F1이 높을 가능성 큼.

---

## 6. 현재 진행 상황

| 단계 | 상태 | 결과 |
|---|---|---|
| EDA + 전처리 설계 | 완료 | StandardScaler 이론 확인, 중복 피처 4개 식별 |
| IF row-level | 완료 | F1 0.5869 (한계 확인) |
| Run-level 집계 도입 | 완료 | F1 0.5869 → 0.8870 (브레이크스루) |
| Autoencoder | 완료 | F1 0.9205 (현재 최고) |
| LOF 실험 | 진행 중 | Standard/52피처 baseline 생성, Standard vs Robust + 52 vs 48 비교 예정 |
| KMeans 실험 | 대기 | LOF 후 |
| KNN run-level | 대기 | 행 수준은 너무 느려 run-level 평균 방식으로 재설계 |
| OC-SVM | 포기 | 선형 커널 한계 + 학습 시간 |
| 앙상블 | 보류 | ML 모델 전부 완료 후 |
| AE 개선 (bottleneck=16) | 보류 | ML 탐색 완료 후 |

---

## 7. 미결 질문 및 가설

- **RobustScaler vs StandardScaler**: LOF/KMeans/KNN에서 어느 쪽이 더 좋은가? → 실험으로 결정
- **피처 제거 (52→48)**: 중복 피처 제거가 거리 기반 모델의 bimodality를 개선하는가? → 실험으로 결정
- **KNN run-level**: run 평균 벡터 500개로 KNN을 구성하면 충분한 성능이 나오는가? → 시도 예정
- **앙상블 구성**: IF + AE + LOF의 점수를 어떻게 결합하면 효과적인가? → ML 완료 후 설계
- **AE bottleneck 크기**: 멘토 제안 16 vs 현재 8. 표현력 vs 과적합 트레이드오프. → 나중에 실험
