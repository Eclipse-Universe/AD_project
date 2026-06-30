"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 49와 동일한 역할(모델 생성+fit).

하이퍼파라미터는 함수 인자로 받을 것 — Exp 1의 contamination 조정이 호출부 한 줄만
바뀌면 되게 하기 위함 (EXPERIMENT_LOG.md Exp 0 다음 계획 참고).
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM


def train_isolation_forest(X: pd.DataFrame, random_state: int, **params) -> IsolationForest:
    # TODO: IsolationForest(random_state=random_state, **params)를 만들어 X에 fit 후 반환
    # params로 contamination 등을 넘길 수 있어야 함
    raise NotImplementedError


def train_sgd_ocsvm(X: pd.DataFrame, random_state: int, **params) -> SGDOneClassSVM:
    # TODO: 백로그 항목 (EXPERIMENT_LOG.md 참고) — 지금 당장 구현하지 않아도 됨
    raise NotImplementedError
