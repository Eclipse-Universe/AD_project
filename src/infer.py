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


def save_submission(pred: np.ndarray, index: pd.Index, path: str) -> None:
    pd.DataFrame(pred, columns=["faultNumber"], index=index).to_csv(path, index=True)
