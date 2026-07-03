"""LOF 피처·스케일러 조합 비교 — 제출 전 로컬 score 분포 확인용.

목적:
  Standard vs Robust, 52피처 vs 48피처(중복 제거) 4가지 조합을 모두 돌려
  run-level score 분포의 bimodality(두 봉우리 분리)가 가장 선명한 조합을 고른다.
  F1을 모르는 상태에서 고르는 것이므로, bimodality + train/test 분리 정도로 판단.

실행: cd src && python run_lof_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_loader import load_test, load_train
from infer import predict_labels_by_run, save_submission
from model import train_lof
from preprocess import (
    fit_scaler,
    scale_features,
    select_features,
    select_features_reduced,
)

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
N_NEIGHBORS = 20

GRID = [
    ("LOF-A", "standard", "full"),
    ("LOF-B", "robust",   "full"),
    ("LOF-C", "standard", "reduced"),
    ("LOF-D", "robust",   "reduced"),
]


def run_one(name, scaler_type, feat_set, train_data, test_data):
    feat_fn = select_features if feat_set == "full" else select_features_reduced

    train_X_raw = feat_fn(train_data)
    test_X_raw  = feat_fn(test_data)

    scaler   = fit_scaler(train_X_raw, scaler_type=scaler_type)
    train_X  = scale_features(train_X_raw, scaler)
    test_X   = scale_features(test_X_raw,  scaler)

    test_run_ids  = test_data["simulationRun"]
    train_run_ids = train_data["simulationRun"]

    lof = train_lof(train_X, n_neighbors=N_NEIGHBORS)

    pred = predict_labels_by_run(
        lof, test_X, test_run_ids,
        run_contamination=RUN_CONTAMINATION, agg="mean",
    )
    pos_rate = pred.mean() if hasattr(pred, "mean") else np.mean(pred)

    # run-level 점수 (낮을수록 이상)
    test_scores  = pd.Series(lof.decision_function(test_X.values), index=test_X.index)
    train_scores = pd.Series(lof.decision_function(train_X.values), index=train_X.index)
    test_run_sc  = test_scores.groupby(test_run_ids.values).mean()
    train_run_sc = train_scores.groupby(train_run_ids.values).mean()

    print(f"\n[{name}] scaler={scaler_type}, feats={feat_set}({len(train_X_raw.columns)}개)")
    print(f"  predicted positive rate : {pos_rate:.4f} (target ~0.322)")
    print(f"  train run scores : mean={train_run_sc.mean():.4f}  std={train_run_sc.std():.4f}  "
          f"min={train_run_sc.min():.4f}  max={train_run_sc.max():.4f}")
    print(f"  test  run scores : mean={test_run_sc.mean():.4f}  std={test_run_sc.std():.4f}  "
          f"min={test_run_sc.min():.4f}  max={test_run_sc.max():.4f}")

    # 분리 지표: (test mean - train mean) / train std — 클수록 이상 run이 train과 멀리 떨어짐
    separation = (train_run_sc.mean() - test_run_sc.mean()) / train_run_sc.std()
    print(f"  separation index (train_mean - test_mean) / train_std = {separation:.3f}")

    save_submission(pred, test_X.index, OUTPUT_DIR / f"output_{name.lower()}.csv")
    return pred, test_run_sc, train_run_sc


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)

    results = {}
    for name, scaler_type, feat_set in GRID:
        pred, test_sc, train_sc = run_one(name, scaler_type, feat_set, train_data, test_data)
        results[name] = (pred, test_sc, train_sc)

    # 시각화: 2×2 서브플롯, 각 조합의 test run score 히스토그램 + train 기준선
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[g[0] for g in GRID],
        shared_xaxes=False,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (name, scaler_type, feat_set), (row, col) in zip(GRID, positions):
        _, test_sc, train_sc = results[name]
        fig.add_trace(go.Histogram(
            x=test_sc.values, name=f"{name} test", opacity=0.65,
            nbinsx=50, marker_color="steelblue",
        ), row=row, col=col)
        fig.add_trace(go.Histogram(
            x=train_sc.values, name=f"{name} train (정상)", opacity=0.65,
            nbinsx=30, marker_color="salmon",
        ), row=row, col=col)

    fig.update_layout(
        title="LOF 4조합 run-level 점수 분포 (steelblue=test, salmon=train정상 / 낮을수록 이상)",
        height=700, width=1100, barmode="overlay", showlegend=False,
    )
    out = EDA_PATH / "lof_grid_distribution.html"
    fig.write_html(str(out))
    print(f"\n시각화 저장: {out}")
    print("\n=== 판단 기준 ===")
    print("1. Test 히스토그램에 두 봉우리가 선명할수록 좋음 (정상 군집 + 이상 꼬리)")
    print("2. Train(정상) 군집이 좁고 오른쪽에 몰려 있을수록 좋음")
    print("3. Separation index가 클수록 정상/이상 run 분리 선명")
    print("4. 위 조건을 가장 잘 만족하는 조합을 제출")


if __name__ == "__main__":
    main()
