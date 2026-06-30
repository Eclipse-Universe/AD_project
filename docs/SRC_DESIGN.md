# src/ 설계

baseline.ipynb의 로직을 그대로 옮기되, 노트북 한 파일에 다 있던 것을 책임 단위로 쪼갠다.
목적은 하나: **실험 하나의 변경이 코드 한 군데의 diff로 드러나게 만드는 것** — 그래야
`docs/EXPERIMENT_LOG.md`의 "변경사항"이 실제로 git log와 1:1로 대응한다.

실제 구현(`.py` 파일 내용)은 직접 작성한다. 아래는 각 파일이 "무엇을 책임지는지"와 그 이유만
정리한 것이다.

## `src/data_loader.py`
- 책임: `train.csv`/`test.csv`를 경로만 받아 `DataFrame`으로 반환. 그 이상의 가공은 하지 않는다.
- 제안 함수: `load_train(path) -> pd.DataFrame`, `load_test(path) -> pd.DataFrame`
- 왜 분리하는가: 데이터 소스가 바뀌거나(예: 로컬 경로 vs 다른 환경) 캐싱을 추가해야 할 때, 다른
  모듈을 전혀 건드리지 않게 하기 위함.

## `src/preprocess.py`
- 책임: baseline의 `process_data()`에 해당. 52개 수치형 컬럼(`xmeas_*`, `xmv_*`)을 **이름으로**
  선택한다.
- 제안 함수: `select_features(df) -> pd.DataFrame`
- 왜 이름으로 선택해야 하는가: `test.csv`는 컬럼이 알파벳순으로 정렬되어 있어 `train.csv`와 순서가
  다르다(PROJECT_CONTEXT.md 참고). 위치(`iloc`) 기반으로 고르면 train/test에서 서로 다른 컬럼을
  뽑는 사고가 날 수 있다.
- 향후 스케일링/피처 추가도 이 파일에 함수를 더하는 식으로 확장한다 (Tier 2 실험과 연결).

## `src/model.py`
- 책임: 모델 생성과 `.fit()`. `contamination` 같은 하이퍼파라미터를 **함수 인자로 받아서**,
  실험마다 값만 바꿔 호출하면 되게 한다 (현재 Exp 1 계획인 contamination 조정이 정확히 이 지점).
- 제안 함수: `train_isolation_forest(X, **params) -> model`, `train_sgd_ocsvm(X, **params) -> model`
  (SGDOneClassSVM도 자리는 만들어 두되, 아직 실행 안 함 — 백로그에 남아있는 항목)

## `src/infer.py`
- 책임: 예측 + baseline cell 53의 라벨 변환(`-1/1` → `1/0`) + 제출 파일 저장.
- 제안 함수: `predict_labels(model, X) -> np.ndarray`, `save_submission(pred, path)`
- 라벨 변환 로직을 여기 한 곳에만 두는 이유: 모델이 늘어나도(IsolationForest, SGDOneClassSVM, 이후
  다른 모델) 변환 규칙은 한 군데서만 관리되게 하기 위함 — 이 변환을 빼먹으면 채점이 통째로 틀어지는
  치명적인 지점이라 더더욱 단일 지점으로 모아둘 필요가 있다.

## `src/run_experiment.py` (entry point)
- 책임: 위 네 모듈을 순서대로 연결(load → select_features → train → predict → save)하고, 실험별
  설정값(모델 종류, `contamination` 등)을 인자로 받는다.
- 왜 필요한가: 이 파일 하나를 실행하는 것이 곧 "실험 하나를 재현하는 것"이 되게 하기 위함. 이
  스크립트 실행 시점의 git commit이 EXPERIMENT_LOG의 해당 항목과 정확히 대응해야 한다.

## 작성 순서 제안
`data_loader.py` → `preprocess.py` → `model.py` → `infer.py` → `run_experiment.py` 순으로, 작은
단위부터 baseline과 결과가 동일한지(F1 재현) 확인하며 올라가는 게 디버깅하기 쉽다.
