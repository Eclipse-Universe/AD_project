"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 45와 동일한 역할(feature 선택).

주의: test.csv는 컬럼이 알파벳순으로 정렬되어 있어 train.csv와 순서가 다르다.
컬럼을 "이름"으로 선택하므로(위치 기반 아님) train/test 양쪽 모두 안전하다.
"""
import pandas as pd

NUMERIC_COLS = [
    'xmeas_1', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
    'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_2',
    'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
    'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_3', 'xmeas_30',
    'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
    'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_4', 'xmeas_40', 'xmeas_41',
    'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmv_1',
    'xmv_10', 'xmv_11', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6',
    'xmv_7', 'xmv_8', 'xmv_9'
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_COLS]


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    diff_df = df.groupby("simulationRun")[NUMERIC_COLS].diff().fillna(0)
    return diff_df.add_suffix("_diff")

