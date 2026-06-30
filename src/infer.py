"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 53, 64와 동일한 역할(예측+저장).

주의: sklearn 이상탐지 모델은 정상=1, 이상=-1을 반환한다.
채점 기준(정상=0, 이상=1)에 맞게 반드시 변환할 것 — 빠뜨리면 채점이 통째로 틀어진다.
"""
import numpy as np
import pandas as pd


def predict_labels(model, X: pd.DataFrame) -> np.ndarray:
    # TODO: model.predict(X) 호출 후 {1: 정상, -1: 이상} -> {0: 정상, 1: 이상}으로 변환
    raise NotImplementedError


def save_submission(pred: np.ndarray, index: pd.Index, path: str) -> None:
    # TODO: "faultNumber" 컬럼으로 DataFrame을 만들어 index=True로 csv 저장 (baseline cell 64)
    raise NotImplementedError
