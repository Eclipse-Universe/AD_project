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

## 5-1. KMeans 실험 발견 (2026-07-03)

### n_clusters 선택 근거
- 너무 작으면 군집이 너무 넓어 이상 run이 정상 군집 내부에 묻힘
- 너무 크면 군집당 샘플 수가 너무 적어 노이즈에 과적합
- 500개 train run 기준 k=50 → 군집당 평균 10개: 군집 안정성과 세밀함의 균형점
- 탐색 범위: k ∈ {10, 20, 30, 50}, 상승 추세 확인 후 50에서 마침

### k값 증가에 따른 패턴
k가 클수록 train 군집 std가 줄어들고 separation index가 높아진다 (70→83).
그러나 어떤 k에서도 이진 예측(237개 이상 run)은 완전히 동일.
→ 이상 run이 정상 군집에서 워낙 멀리 떨어져 있어, 군집 구조의 세밀함과 무관하게 항상 상위 32%에 속함.

### KMeans vs LOF 관계
- LOF: 로컬 밀도 비교 → 정상 군집에서 국소적으로 이탈한 이상 감지
- KMeans: 글로벌 군집 중심 거리 → 정상 운전 패턴 어디에도 속하지 않는 이상 감지
- 두 모델이 694/740 run에서 동의, 46개 run(각 23개)에서 의견 다름
- KMeans separation index(82.98) > LOF(68.57) → 분리가 더 선명하지만 동일한 run 집합을 판정

### KMeans Euclidean vs Mahalanobis — 핵심 차이

KMeans-Euclid (Exp 16): F1 0.9083, Recall 0.8969, FN ~25 runs
KMeans-Mahal (Exp 17): 실험 진행 중

Euclidean이 실패한 구체적 패턴 (형태 B 이상):
- 정상: xmv_7 ≈ xmeas_12 (r=1.0 공선관계 유지)
- 이상: xmv_7=+1.5σ, xmeas_12=−0.5σ (관계 파괴, 개별값은 moderate)
- Euclidean: 두 피처가 각각 군집 중심에서 적당히 가까워 낮은 이상 점수 → FN
- Mahalanobis: Σ⁻¹이 "xmv_7과 xmeas_12가 반대 방향 = 상관 파괴"를 포착 → 높은 이상 점수

### GitHub 공개/비공개 재구성 (2026-07-03)

**공개 (GitHub)**:
- src/ 전체 (소스 코드)
- docs/EXPERIMENT_LOG.md, ENGINEERING_LOG.md, EDA_SUMMARY.md, SRC_DESIGN.md, CONCEPTS.md
- eda/ 스크립트 및 CSV
- README.md, PROJECT_CONTEXT.md, baseline_code/

**비공개 (.gitignore에 추가, 로컬 보존)**:
- docs/MENTORING_QUESTIONS.md — 개인 멘토링 Q&A
- docs/QA_LOG.md — 내부 의사결정 대화 기록
이유: 경쟁 전략 정보, 개인 학습 과정이 포함되어 있어 공개 포트폴리오에 적합하지 않음.

### 출력 파일 명명 규칙 (2026-07-03 확정)
`output_exp{순번}({모델명}).csv` 형식으로 통일.
기존 파일 모두 소급 적용. 그리드 탐색용 임시 파일은 제출 후 삭제.
- IF 계열: output_exp0~4(IF), exp8~12(IF-run), exp14/14b(IF)
- OC-SVM: exp5~7(OC-SVM)  
- AE: exp13(AE)
- LOF: exp15(LOF)
- KMeans: exp16(KMeans)

---

## 6. 현재 진행 상황

| 단계 | 상태 | 결과 |
|---|---|---|
| EDA + 전처리 설계 | 완료 | StandardScaler 이론 확인, 중복 피처 4개 식별 |
| IF row-level | 완료 | F1 0.5869 (한계 확인) |
| Run-level 집계 도입 | 완료 | F1 0.5869 → 0.8870 (브레이크스루) |
| Autoencoder | 완료 | F1 0.9205 (현재 최고) |
| LOF 실험 | 완료 | LOF-A(Standard+52피처) F1 0.9237, 전체 최고 |
| KMeans-Euclid 실험 | 완료 | F1 0.9083 — 피처 독립 가정으로 형태 B 이상 미포착 |
| KMeans-Mahal 실험 | 완료 | F1 0.9277, **전체 최고점** — 상관 파괴 이상 포착 |
| KNN run-level | 완료 | k=5, sep=137, LOF/Mahal과 89% 일치 (앙상블 타이브레이커 역할 예상) |
| OC-SVM | 포기 | 선형 커널 한계 + 학습 시간 |
| 앙상블 | 보류 | ML 모델 전부 완료 후 |
| AE 개선 (bottleneck=16) | 보류 | ML 탐색 완료 후 |

---

## 7. 미결 질문 및 가설

- **RobustScaler vs StandardScaler**: LOF/KMeans/KNN에서 어느 쪽이 더 좋은가? → 실험으로 결정
- **피처 제거 (52→48)**: 중복 피처 제거가 거리 기반 모델의 bimodality를 개선하는가? → 실험으로 결정
- **KNN run-level 앙상블 가치**: LOF·Mahal과 89% 일치 (vs 두 모델 상호 93.8% 일치). KNN만의 40개 플래그는 다수결에서 자동 제외 → 품질 리스크 낮음. 가치는 LOF vs Mahal 불일치 46개 run에서 타이브레이커 역할.
- **앙상블 구성**: LOF + KMeans-Mahal + KNN 점수 결합. 각 모델의 강점이 다름:
  - LOF: 로컬 밀도 기반 (Precision 안정적)
  - KMeans-Mahal: 글로벌 공분산 보정 거리 (Recall 최고)
  - KNN: k-이웃 거리 기반 (다른 관점)
  → 이론적 천장 0.93+ 기대
- **AE bottleneck 크기**: 멘토 제안 16 vs 현재 8 → ML 완료 후 실험
- **RUN_CONTAMINATION fine-tuning for Mahal**: Recall(0.9361) > Precision(0.9195) 격차 0.017
  → 0.31 시도 여지 있으나 현재 F1이 최고점이므로 KNN 먼저

### 마할라노비스 거리의 성공 요인 (확정)

Exp 17 결과(F1 0.9277)로 다음이 실험적으로 증명됨:
1. **상관 파괴 신호 포착**: 개별 피처값이 평범해도 피처 간 관계가 깨지면 마할라노비스 거리가
   크게 증가 → 유클리드 거리가 놓친 형태 B 이상(FN ~25→~15) 수정
2. **분산 지배 문제 해소**: xmv_5(std_ratio 18.46) 같은 고분산 피처가 거리를 지배하는
   것을 Σ⁻¹이 자동 정규화 → 잘못된 오탐(FP) 일부 수정
3. **EDA 수치와 이론의 정합성**: |왜도|<0.2 → 다변량 정규분포 → 마할라노비스 최적.
   EDA 결과가 모델 선택에 직접 기여한 케이스.
