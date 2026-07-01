# 멘토링 질문 리스트

프로젝트를 진행하며 우리끼리 경험적으로 판단했거나, 확신 없이 가정하고 넘어간 지점들을 모았다.
각 질문에 "왜 이게 궁금한지" 맥락을 같이 적어, 멘토가 배경 설명 없이 바로 답할 수 있게 한다.

---

## A. 도메인(화학 공정/TEP) 관련

1. **TEP는 워낙 유명한 벤치마크라 기존 연구가 많을 텐데, row-level Isolation Forest 같은 일반
   tabular 이상탐지 방법이 보통 어느 정도 성능(F1, 또는 fault별 detection rate)을 내는지, 그리고
   화학공정 분야 고전 기법(PCA 기반 T²/SPE 통계, PLS 등)과 비교했을 때 우리가 맞는 방향으로 가고
   있는지 궁금하다.** (배경: threshold 튜닝만으로 F1 0.58 근처에서 막혔고 diff feature 실험 중,
   `docs/EXPERIMENT_LOG.md` Exp 0~3 참고)

2. **train 데이터(500개 run)가 "정상"의 모든 운전 영역을 대표한다고 가정해도 되는지, 아니면
   실제 공정에는 생산 grade 변경 등으로 여러 "정상 모드"가 존재해서 단순 정상/이상 이분법이
   위험할 수 있는지** 궁금하다. (배경: 다중 정상 모드가 있다면 일부 정상 패턴을 이상으로 오탐할
   위험이 있음)

3. **실제 화학공정 센서 데이터의 노이즈 특성(샘플링 주기, 측정 오차 등)이 이 시뮬레이션 데이터와
   얼마나 비슷한지, 실무에서는 저역통과 필터 같은 전처리가 표준적으로 들어가는지** 궁금하다.

---

## B. 방법론(검증/threshold) 관련

4. **train이 100% 정상이라 로컬에서 recall을 측정할 방법이 없어, 매 실험마다 리더보드 제출에
   의존하고 있다.** 이런 구조(novelty detection, 정상만 있는 train)의 대회/프로젝트에서 표준적으로
   쓰는 로컬 검증 기법(예: synthetic fault injection, 정상 데이터에 인위적 변형을 가해 가짜 이상치를
   만드는 방법)이 있는지 궁금하다.

5. ~~**`IsolationForest`의 `contamination`을 threshold 다이얼로 쓰는 게 실무 관행인지**~~ →
   **[경험적으로 유효함 확인, 질문 전환]** Exp 0~2에서 contamination을 auto → 0.32 → 0.23으로
   조정해가며 F1 곡선이 역U자임을 실제로 확인했다. 다만 `decision_function()` 점수에 직접
   임계값을 적용하는 방식과 비교했을 때 어떤 차이가 있는지, 그리고 우리처럼 contamination을
   threshold 다이얼로 쓰는 것이 대회/실무에서 실제로 허용되는 패턴인지는 여전히 궁금하다.

6. **accuracy + F1 + 예측 positive 개수만으로 confusion matrix(TP/FP/FN/TN)를 역산하는 기법을
   Exp 0~3에서 계속 쓰고 있다.** 수학적으로는 검산까지 맞아떨어졌지만, 입력값(accuracy/F1)이
   반올림된 값이라 실험이 누적될수록 오차가 커지지는 않는지, 더 신뢰할 수 있는 방법이 있는지
   궁금하다.

---

## C. 모델링/구조 관련

7. **라벨이 `simulationRun` 단위로 통째로 0 또는 1인데(`PROJECT_CONTEXT.md` 참고), 우리는 지금
   row를 독립적으로 다루고 있다.** 이 구조에서는 row-level 분류보다 run(시퀀스) 단위 분류나 row
   score를 run 단위로 집계하는 방식이 정석인지, 멘토 의견이 궁금하다(현재 Tier 3 후보로만 남겨둔
   상태, `docs/EXPERIMENT_LOG.md` 백로그 참고).

8. **scikit-learn IsolationForest만 계속 쓰고 있는데**, 공정 변수 간 강한 선형/비선형 상관관계가
   있는 이 데이터 특성상 거리 기반 모델(One-Class SVM, LOF)이나 오토인코더 재구성 오차 기반 방법이
   실제로 더 잘 맞을 가능성이 있는지 궁금하다. (현재 계획: IF 한계 확인 후 SGDOneClassSVM → 이후
   Autoencoder 순으로 실험 예정 — 한 번에 하나씩 바꿔 원인을 분리하기 위함)

9. **diff feature(직전 시점 대비 변화량)를 추가해 IF에 시간 정보를 주는 실험(Exp 3)을 진행 중이다.**
   - diff가 효과가 있다면: "고장 초반 점진적 드리프트를 변화량 feature가 잡았다"는 가설이 맞는 것 —
     그렇다면 rolling mean/std나 window 기반 feature까지 추가하는 게 의미 있는지?
   - diff가 효과가 없다면: "row 단위 feature 자체로는 한계가 있고, 시퀀스 전체를 보는 모델이 필요하다"
     는 신호 — 그게 오토인코더나 LSTM-AE로 넘어가야 하는 시점인지?
   - FDC(Fault Detection and Classification) 분야에서 diff보다 더 일반적으로 쓰는 시간 feature가
     있다면 알고 싶다.

10. **실험을 설계할 때 "한 번에 하나씩만 바꾼다"(feature 변경과 모델 변경을 분리)는 원칙으로
    진행하고 있다.** 이게 ML 실험 방법론 관점에서 올바른 접근인지, 아니면 더 체계적인 방법(예:
    ablation study, grid search처럼 여러 조합을 한 번에 비교하는 방식)이 실무/대회에서 더 많이
    쓰이는지 궁금하다. (배경: 현재 방식은 원인 분리가 명확하다는 장점이 있지만, 실험 횟수가 선형으로
    늘어나는 단점이 있음)

11. **SGDOneClassSVM을 row-level로 실험했을 때 F1이 0.38로 낮았다(Exp 5~7). 이후 IsolationForest에
    run-level 집계(run별 decision_function 평균)를 적용하니 F1이 0.87로 급등했다(Exp 8~12).** 같은
    run-level 집계 방식을 SGDOneClassSVM에 적용하면 성능이 개선될 수 있을까? "선형 커널이라 근본적으로
    한계가 있다"고 판단해 시도하지 않았는데, row-level 점수 품질이 나쁘더라도 960개 평균을 냈을 때
    이상/정상 run이 분리될 가능성이 있는지, 그리고 그 가능성을 사전에 판단하는 방법이 있는지 궁금하다.

---

(멘토링 후 답변·결정 사항은 여기에 추가)
