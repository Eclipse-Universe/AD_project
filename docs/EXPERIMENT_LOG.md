# Experiment Log

리더보드에 제출할 때마다 아래 형식으로 기록한다: 변경사항 → 가설(이유) → 점수 → 분석 → 다음 계획.
"왜 이 점수가 나왔는가"에 대한 근거가 핵심이며, 합의 없이 다음 실험으로 넘어가지 않는다.

---

## Exp 0 — Baseline (IsolationForest, default params)

**날짜**: 2026-06-30

**재현 커밋**: `5f60511` — `cd src && python run_experiment.py` (EXP_NAME="exp0",
CONTAMINATION="auto"). 이 커밋의 코드로 실행하면 `outputs/output_exp0.csv`가
`baseline_code/output.csv`와 행 단위로 완전히 동일함을 확인함 (710,400행 전수 비교).

**변경 사항**: `baseline_code/baseline.ipynb` 그대로 실행.
- Feature: `xmeas_1~41`, `xmv_1~11` (52개), 스케일링 없음, `simulationRun`/`sample` 제외
- Model: `IsolationForest(random_state=42)` (sklearn 기본 파라미터, `contamination='auto'`)
- Validation: 없음 (train에 정상 데이터만 있어 hold-out 불가)
- 코드 리뷰 메모: cell 50에서 `train()` 함수로 IsolationForest를 학습했지만, cell 51에서 동일한
  설정의 IsolationForest를 다시 만들어 덮어쓴다. 결과적으로 `SGDOneClassSVM`은 import·설명만 되고
  실제 제출 결과에는 전혀 관여하지 않았다.

**점수 (public)**: F1 0.5607, Accuracy 0.7936

**Confusion matrix 역산** (정답 라벨 없이도 accuracy + F1 + 예측 positive 개수(`output.csv`에서
직접 카운트, PP=105,204/710,400)로 4칸을 모두 풀어낼 수 있음 — 유도: `FP+FN = N(1-accuracy)`,
이를 F1 식에 대입하면 TP에 대한 1차방정식이 되어 풀림. accuracy/F1 역산값과 목표값이 소수점 4자리
까지 일치해 신뢰 가능):

| | 예측 normal | 예측 anomaly |
|---|---|---|
| 실제 normal | TN ≈ 470,200 | FP ≈ 11,631 |
| 실제 anomaly | FN ≈ 134,996 | TP ≈ 93,573 |

→ Recall ≈ 0.4094, Precision ≈ 0.8894. 모델이 예측한 positive 비율은 14.8%인데 위 식으로 추정한
실제 anomaly 비율은 약 32.2% — **모델이 실제보다 훨씬 적게 "이상"이라고 판단하고 있음(과소 탐지)**.

**분석 (확정)**:
1. **Threshold 불일치 (확인됨)**: `contamination='auto'`(약 10~15% 가정)가 실제 이상 비율(약 32%)
   보다 훨씬 낮음 → precision은 높지만(89%) recall이 크게 희생(41%). 가장 직접적인 원인.
2. **시간 정보 미사용 (보조 원인으로 추정)**: 문제 정의상 한 `simulationRun`은 통째로 0 또는 1인데,
   모델은 row를 독립적으로 봄. 고장 초반의 점진적 드리프트 구간 row는 정상처럼 보일 수 있어 FN이
   커지는 데 기여했을 가능성 — threshold 조정만으로는 한계가 있을 수 있음. (아직 직접 검증 안 됨)
3. Train/Test 분포 차이, 스케일 미정규화: 위 1, 2번 대비 영향이 작을 것으로 추정 — 우선순위 낮음.

**다음 시도 (합의됨 — Tier 1부터 진행)**:
- [x] confusion matrix 역산으로 FP vs FN 비대칭 확인 → FN이 압도적으로 큼(과소 탐지)
- [x] **Exp 1**: `contamination`을 실제 추정치(~0.32)에 맞게 조정 → 아래 항목 참고
- [ ] (Tier 2, 보류) run 내 rolling mean/std, 직전 시점 대비 diff 등 시간창 feature 추가
- [ ] (Tier 3, 보류) row-level score를 run 단위로 집계해 run 전체를 분류, 또는 시퀀스 모델(LSTM-AE 등)
- [ ] SGDOneClassSVM을 실제로 학습/제출해 IsolationForest와 비교 (현재는 미실행 상태, 우선순위 낮음)

---

## Exp 1 — contamination 0.32 (`'auto'` → 0.32)

**날짜**: 2026-06-30

**재현 커밋**: `d2327fe` — `cd src && python run_experiment.py` (EXP_NAME="exp1", CONTAMINATION=0.32)

**변경 사항**: Exp 0과 모든 게 동일, `IsolationForest`의 `contamination`만 `'auto'` → `0.32`로
변경. 가설: train 기준 threshold를 풀어주면 test에서 더 많이 "이상"으로 잡아 recall이 오를 것이다
(Exp 0 분석의 직접적인 후속).

**로컬 진단 (제출 전)**: predicted positive rate가 14.81% → **44.36%**로 상승 — 목표였던 32.2%를
12%p 초과. `contamination`이 train(100% 정상) 점수 분포 기준 threshold이기 때문에, 그 threshold를
test에 적용했을 때 정확히 같은 비율이 나온다는 보장이 없다는 점(Q&A 기록 참고)이 그대로 드러남.
test 점수 분포가 train보다 더 넓게 퍼져 있다는 뜻이며, 두 가지 가능성을 구분해야 함:
1. test의 실제 이상 run들이 점수를 크게 끌어내림 (recall이 크게 오르는 좋은 신호)
2. test의 정상 run도 train 정상과 미세하게 달라 threshold에 과민 반응 (precision이 크게 깎이는
   나쁜 신호)

**점수 (public)**: F1 0.5699, Accuracy 0.6704

**Confusion matrix 역산** (Exp 0와 동일한 방법, PP=315,112):

| | Exp 0 (`auto`) | Exp 1 (`0.32`) |
|---|---|---|
| TP | 93,573 | 155,128 |
| FP | 11,631 | 159,984 |
| FN | 134,996 | 74,164 |
| TN | 470,200 | 321,124 |
| Recall | 0.4094 | 0.6766 |
| Precision | 0.8894 | 0.4923 |
| F1 | 0.5607 | 0.5699 |
| Accuracy | 0.7936 | 0.6704 |
| 추정 실제 anomaly 비율 | 0.3217 | 0.3228 (두 실험이 독립적으로 거의 같은 추정치를 줌 — 신뢰도 ↑) |

**분석**:
1. **threshold 위치가 핵심 병목이 아니었다.** FP가 11,631 → 159,984로 13배 넘게 폭증하고 recall이
   41%→68%로 크게 올랐는데도 F1은 +0.0092밖에 안 올랐다 — recall/precision이 거의 1:1로 상쇄되며
   F1 곡선이 이 구간에서 평평하다는 뜻.
2. **모델의 anomaly score 자체의 분리력이 약하다는 신호.** predicted positive rate(44.36%)가 이미
   추정 실제 비율(32.3%)을 넘어섰는데도 recall이 67.66%에 그쳤다 — score가 정상/이상을 잘
   구분했다면 실제 비율 근처로 threshold를 맞췄을 때 recall이 100%에 가까워야 한다. 그렇지 않다는
   건 새로 잡힌 FP 다수가 진짜 이상이 아니라 그냥 threshold 근처의 애매한 정상 데이터라는 뜻 —
   Exp 0에서 보류했던 가설 2번(row 독립 가정의 한계, 시간 정보 미사용)이 이 결과로 뒷받침됨.
3. 결론: Tier 1(threshold 튜닝)만으로는 한계에 가까워 보인다. 한 지점(0.32)만으로 단정하기엔
   이르지만, recall/precision이 거의 대칭적으로 상쇄된 점이 강한 신호.

---

## Exp 2 — contamination 0.23 (Exp 1 결과로부터 외삽)

**날짜**: 2026-06-30

**재현 커밋**: `82837ad` — `cd src && python run_experiment.py` (EXP_NAME="exp2", CONTAMINATION=0.23)

**변경 사항**: Exp 1의 predicted positive rate(44.36%)가 목표(32.2%)를 1.39배 초과했으므로,
`0.32 / 1.39 ≈ 0.23`으로 거칠게 외삽해 다음 값을 잡았다. Tier 1(threshold 튜닝)이 한계에
부딪혔는지, 아니면 아직 더 나은 지점이 남아있는지(F1 곡선의 모양)를 확인하기 위한 세 번째 점.

**로컬 진단 (제출 전)**: predicted positive rate 36.13% — 목표(32.2%)와의 오차가 Exp 1(12%p)보다
줄어든 4%p. 외삽 방향은 맞았지만 여전히 정확히 일치하지는 않아 contamination↔test rate 관계가
완전한 선형은 아님을 다시 확인.

**점수 (public)**: _(제출 후 기록 예정)_

**분석**: _(점수 받으면 Exp 0/1/2 F1을 함께 비교해 곡선 모양 판단)_

---

(다음 실험은 여기에 이어서 추가)
