"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 53, 64와 동일한 역할(예측+저장).

주의: sklearn 이상탐지 모델은 정상=1, 이상=-1을 반환한다.
채점 기준(정상=0, 이상=1)에 맞게 변환한다 — 이걸 빠뜨리면 채점이 통째로 틀어진다.
"""
import numpy as np
import pandas as pd


def predict_labels(model, X: pd.DataFrame) -> np.ndarray:
    pred_y = model.predict(X)
    if -1 in pred_y:
        pred_y = (pred_y == -1).astype(int)
    return pred_y


def predict_labels_by_run(model, X: pd.DataFrame, run_ids: pd.Series, run_contamination: float) -> np.ndarray:
    # decision_function: 낮을수록 이상 (IF 기준)
    scores = model.decision_function(X)

    # run별 평균 점수 계산
    run_mean_scores = pd.Series(scores, index=X.index).groupby(run_ids.values).mean()

    # 하위 run_contamination 비율의 run을 이상으로 판정
    threshold = run_mean_scores.quantile(run_contamination)
    anomalous_runs = run_mean_scores.index[run_mean_scores <= threshold]

    # row 단위로 확장 (같은 run의 모든 row에 동일 라벨 적용)
    return run_ids.isin(anomalous_runs).astype(int).values


def save_submission(pred: np.ndarray, index: pd.Index, path: str) -> None:
    pd.DataFrame(pred, columns=["faultNumber"], index=index).to_csv(path, index=True)
