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


def predict_labels_by_run(
    model,
    X: pd.DataFrame,
    run_ids: pd.Series,
    run_contamination: float,
    agg: str = "mean",
    score_percentile: int = 10,
) -> np.ndarray:
    # decision_function: 낮을수록 이상 (IF 기준)
    scores = pd.Series(model.decision_function(X), index=X.index)
    grouped = scores.groupby(run_ids.values)

    if agg == "mean":
        run_scores = grouped.mean()
    elif agg == "percentile":
        # 하위 score_percentile% — run 안에서 가장 이상다운 순간들을 대표값으로 사용
        # min보다 잡음에 강하고, mean보다 고장 발전 후반부를 더 잘 반영
        run_scores = grouped.quantile(score_percentile / 100)
    elif agg == "min":
        run_scores = grouped.min()

    threshold = run_scores.quantile(run_contamination)
    anomalous_runs = run_scores.index[run_scores <= threshold]
    return run_ids.isin(anomalous_runs).astype(int).values


def save_submission(pred: np.ndarray, index: pd.Index, path: str) -> None:
    pd.DataFrame(pred, columns=["faultNumber"], index=index).to_csv(path, index=True)
