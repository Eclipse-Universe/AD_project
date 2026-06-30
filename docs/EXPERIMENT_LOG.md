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
- [ ] **Exp 1 예정**: `contamination`을 실제 추정치(~0.32)에 맞게 조정 — 가장 작은 변화로 가설 1을
      직접 검증. `src/` 모듈 구조로 포팅하며 진행.
- [ ] (Tier 2, 보류) run 내 rolling mean/std, 직전 시점 대비 diff 등 시간창 feature 추가
- [ ] (Tier 3, 보류) row-level score를 run 단위로 집계해 run 전체를 분류, 또는 시퀀스 모델(LSTM-AE 등)
- [ ] SGDOneClassSVM을 실제로 학습/제출해 IsolationForest와 비교 (현재는 미실행 상태, 우선순위 낮음)

---

(다음 실험은 여기에 이어서 추가)
