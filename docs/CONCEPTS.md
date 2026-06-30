# 사전 개념 정리 (Prerequisite Concepts)

이 문서는 `baseline_code/baseline.ipynb`의 각 단계와 직접 연결되는 이론적 배경을 정리합니다.
나중에 포트폴리오나 발표에서 "왜 이렇게 했는가"에 답할 수 있도록, 각 개념이 코드의 어느 지점과
맞닿아 있는지 명시합니다.

## 1. 문제 정의 — Novelty Detection (One-Class Classification)
- 일반 지도학습 분류와 다른 점: train에 negative class(이상 사례)가 전혀 없음.
- 학계에서는 이를 **Novelty Detection**(정상만 보고 학습 후, "낯선" 패턴 탐지)으로 분류하며, train
  자체에 오염된 이상치가 섞여있는 **Outlier Detection**과 구분한다. scikit-learn 문서도 이 둘을
  `novelty=True/False` 옵션 등으로 구분해서 설명한다.
- → baseline.ipynb 5절(Model Training)에서 `IsolationForest`/`SGDOneClassSVM`을 정상 데이터에만
  `.fit()`하는 이유가 여기서 나온다.

## 2. 도메인 지식 — Tennessee Eastman Process (TEP)
- 1993년 Downs & Vogel이 제안한 화학공정 고장진단(FDC) 연구용 표준 시뮬레이터.
- 원본 TEP 데이터셋은 IDV(1)~IDV(20)의 21가지 fault 유형(멀티클래스)을 갖고 있으나, 본 대회는
  이를 정상(0)/이상(1) **이진화**한 버전이다. → "이상"이라는 라벨 뒤에는 여러 종류의 서로 다른
  고장 메커니즘이 섞여 있을 수 있다는 뜻이며, 단일 결정경계로 잡기 어려운 이질적 패턴일 가능성을
  시사한다.
- `xmeas_*`(41개): 온도, 압력, 유량, 액위, 조성(가스크로마토그래피) 등 실측 센서값.
- `xmv_*`(11개): 밸브/제어 신호. 공정 제어 루프가 측정값을 보고 일부 `xmv`를 능동적으로 조정하므로,
  `xmeas`와 `xmv` 사이에는 **제어 루프로 인한 인과적 상관관계**가 존재한다 (baseline 3.3절에서 본
  `xmeas_12`-`xmv_7` 상관계수 ≈1이 그 예).
- → 이상이 발생하면 제어 루프가 보상 동작을 하면서 평소엔 강하게 상관된 변수 쌍의 관계가 깨질 수
  있다는 baseline의 "고민해볼 사항"(cell 42)이 바로 이 제어 루프 구조에서 나온다.

## 3. 평가지표 — F1-score
- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = precision·recall의 조화평균.
- Accuracy 대신 F1을 쓰는 이유: 정상/이상 클래스 비율이 50:50이 아닐 가능성이 높고, 단순 정확도는
  다수 클래스만 맞혀도 높게 나오는 함정이 있다.
- One-Class 모델은 기본적으로 "이상치 비율을 얼마로 가정하느냐"(threshold)에 따라 precision/recall이
  trade-off 된다 → `IsolationForest`의 `contamination` 파라미터, `decision_function()` 점수에 대한
  임계값 선택이 F1에 직접 영향을 준다.

## 4. 모델 1 — Isolation Forest
- 핵심 아이디어: 무작위로 변수를 고르고 무작위 분할점으로 데이터를 반복 분할하는 트리(iTree)를 다수
  생성. 이상치는 정상 데이터보다 **적은 분할 횟수로 고립**되는 경향(짧은 path length) → 평균 path
  length가 짧을수록 이상치 점수가 높음.
- 거리 기반이 아니라 분할 기반이라 **스케일에 비교적 덜 민감**하지만, 변수마다 분산/range가 크게
  다르면 무작위 분할점이 어떤 변수에서는 항상 비효율적으로 잡힐 수 있어 완전히 무관하지는 않다.
- `contamination` 기본값은 `'auto'`(score offset -0.5 기준) — 실제 test의 이상 비율과 다를 수 있음
  → F1에 영향.

## 5. 모델 2 — One-Class SVM / SGDOneClassSVM
- 원조 One-Class SVM(Schölkopf, 2001): 커널 공간에서 정상 데이터를 원점으로부터 최대 마진으로
  분리하는 초평면을 찾는다. `nu` 파라미터가 이상치 비율 상한의 추정치 역할.
- `SGDOneClassSVM`은 선형 커널을 SGD(확률적 경사하강법)로 근사한 버전 — 대용량 데이터에 빠르지만
  **비선형 관계 포착 능력이 떨어진다.** 공정 변수 간 관계가 비선형일 가능성(센서 포화, 비선형 제어
  응답 등)을 고려하면 한계가 있을 수 있다.
- **거리/마진 기반 모델은 스케일에 매우 민감** — baseline은 정규화 없이 바로 fit하므로, 분산이 큰
  변수(e.g. `xmeas_7` ~2700대, `xmeas_4` ~9대)가 거리 계산을 지배해버릴 가능성이 있다.
- 코드 리뷰 메모: baseline.ipynb cell 51에서 IsolationForest가 다시 fit되어 cell 50의 `model`을
  덮어쓰므로, 실제 제출(F1 0.5607)에는 SGDOneClassSVM이 전혀 관여하지 않았다 (import·설명만 되고
  학습/제출되지 않음).

## 6. 전처리 일반론 — Feature Scaling
- StandardScaler/MinMaxScaler 등으로 변수 스케일을 맞추는 이유: 트리 기반 모델은 영향이 적지만,
  거리/커널 기반 모델(OneClassSVM)은 스케일 차이가 곧 "어떤 변수가 더 중요하게 취급되는가"를
  결정해버린다.
- baseline `process_data()`(cell 45)는 스케일링을 전혀 하지 않는다 → 의도적으로 비워둔 부분
  ("정규화와 feature engineering 과정은 포함되지 않았다", cell 44).

## 7. 검증(Validation) 전략의 어려움
- 일반적인 분류라면 train을 정상/비정상 모두 포함한 채로 hold-out 하면 되지만, 이 데이터는 train에
  비정상이 아예 없어 같은 방식이 불가능하다.
- baseline은 validation을 생략하고 바로 test/리더보드 점수로 판단한다(cell 48에 명시) → **public
  리더보드에 과적합할 위험**이 있다 (반복 제출로 public set 특성에 맞춰버리는 것).
- 대안: train 내 일부 simulationRun을 정상 hold-out으로 분리해 "정상을 정상으로 잘 보존하는지"
  (false positive rate)만이라도 사전 점검하는 방법 등을 논의해볼 수 있다.

## 8. 시계열 구조 — Row Independence 가정의 한계
- 각 `simulationRun`은 시간 순으로 정렬된 시계열(train 500 step, test 960 step)이지만, baseline은
  모든 row를 독립적인 i.i.d. 샘플처럼 다룬다.
- 화학 공정의 이상은 보통 **점진적 드리프트**나 **여러 변수의 동시다발적 패턴 변화**로 나타나는
  경우가 많아, 단일 시점(row) 값만으로는 정상 범위 안에 있어 보여도 이상일 수 있다.
- 개선 방향(논의 필요): rolling mean/std 같은 윈도우 feature, 인접 시점과의 변화량(diff), 혹은
  시퀀스 모델(LSTM-AE, Transformer 기반 reconstruction 등)으로 시간 축 정보를 명시적으로 사용하는
  방법이 있다 — 단, baseline 이해 및 직접 구현 우선이므로 이건 추후 논의.

## 부록 — baseline.ipynb 섹션 매핑
| baseline 절 | 개념 |
|---|---|
| 3.1 데이터 유형 확인 | 변수 분포, fault label 정의 |
| 3.2~3.3 Correlation | 제어 루프 기반 변수 간 인과적 상관관계 |
| 4. Data Process | feature selection, scaling 부재 |
| 5. Model Training | Novelty detection, Isolation Forest / One-Class SVM 원리 |
| 6. Model Inference | label 변환(-1/1→0/1), F1 평가의 threshold 민감성 |
