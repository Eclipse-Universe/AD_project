# Experiment Log

리더보드에 제출할 때마다 아래 형식으로 기록한다: 변경사항 → 가설(이유) → 점수 → 분석 → 다음 계획.
"왜 이 점수가 나왔는가"에 대한 근거가 핵심이며, 합의 없이 다음 실험으로 넘어가지 않는다.

---

## Exp 0 — Baseline (IsolationForest, default params)

**날짜**: 2026-06-30

**변경 사항**: `baseline_code/baseline.ipynb` 그대로 실행.
- Feature: `xmeas_1~41`, `xmv_1~11` (52개), 스케일링 없음, `simulationRun`/`sample` 제외
- Model: `IsolationForest(random_state=42)` (sklearn 기본 파라미터, `contamination='auto'`)
- Validation: 없음 (train에 정상 데이터만 있어 hold-out 불가)
- 코드 리뷰 메모: cell 50에서 `train()` 함수로 IsolationForest를 학습했지만, cell 51에서 동일한
  설정의 IsolationForest를 다시 만들어 덮어쓴다. 결과적으로 `SGDOneClassSVM`은 import·설명만 되고
  실제 제출 결과에는 전혀 관여하지 않았다.

**점수 (public)**: F1 0.5607

**분석 (가설)**:
1. **Threshold 불일치**: `contamination='auto'`가 가정하는 이상치 비율이 실제 test set의 이상
   비율과 다를 가능성 — precision/recall 중 한쪽이 크게 떨어졌을 수 있다 (어느 쪽인지는 confusion
   matrix를 직접 봐야 확정 가능, 아직 미확인).
2. **시간 정보 미사용**: row 단위 독립 가정으로 인해, 점진적으로 발생하는 이상(서서히 정상 범위를
   벗어나는 패턴)을 단일 시점 값만으로는 포착하지 못했을 가능성.
3. **Train/Test 분포 차이**: train은 run당 500 step, test는 960 step — test에 다른 길이의 transient
   구간이 포함되어 있어 "정상"의 분포 자체가 약간 다를 수 있음.
4. **스케일 미정규화**: IsolationForest는 스케일에 비교적 둔감하므로 이번 영향은 제한적일 것으로
   추정(우선순위 낮음).

**다음 시도 후보 (논의 후 진행 예정 — 아직 미결정)**:
- [ ] confusion matrix로 어느 쪽 오류(FP vs FN)가 큰지 먼저 확인 → 이유 기반 우선순위 결정
- [ ] `contamination` 값 조정 또는 `decision_function()` 점수 분포 직접 분석 후 threshold 튜닝
- [ ] SGDOneClassSVM을 실제로 학습/제출해 IsolationForest와 비교 (현재는 미실행 상태)
- [ ] 스케일링 적용 후 OneClassSVM 계열 재시도

---

(다음 실험은 여기에 이어서 추가)
