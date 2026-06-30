"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 45와 동일한 역할(feature 선택).

주의: test.csv는 컬럼이 알파벳순으로 정렬되어 있어 train.csv와 순서가 다르다.
반드시 컬럼을 "이름"으로 선택할 것 — 위치(iloc) 기반 선택 금지.
"""
import pandas as pd

NUMERIC_COLS = [
    # TODO: baseline.ipynb cell 45의 52개 xmeas_*/xmv_* 컬럼 이름을 그대로 옮겨올 것
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: NUMERIC_COLS만 선택해 반환
    raise NotImplementedError
