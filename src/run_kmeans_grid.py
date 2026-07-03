"""KMeans 이상 탐지 — n_clusters 탐색 + 스케일러 비교.

LOF 실험에서 확정된 사항:
  - 52개 피처 고정 (48개로 줄이면 이상 시 상관 파괴 신호를 잃음)
  - StandardScaler 우선 (|왜도|<0.2, LOF에서 Robust와 차이 미미)

KMeans에서 새로운 변수: n_clusters
  - 너무 작으면 군집이 너무 넓어 이상 run이 내부에 묻힘
  - 너무 크면 노이즈에 맞춰진 군집이 정상 run을 오탐
  - {10, 20, 30, 50} 4가지 탐색 → 최적 k로 Robust도 비교

실행: cd src && python run_kmeans_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_loader import load_test, load_train
from infer import predict_labels_by_run, save_submission
from model import KMeansAnomalyDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

K_VALUES = [10, 20, 30, 50]


def run_one(name, model, train_X, test_X, test_run_ids, train_run_ids):
    pred = predict_labels_by_run(
        model, test_X, test_run_ids,
        run_contamination=RUN_CONTAMINATION, agg="mean",
    )
    pos_rate = np.mean(pred)

    test_sc  = pd.Series(model.decision_function(test_X.values),  index=test_X.index)
    train_sc = pd.Series(model.decision_function(train_X.values), index=train_X.index)
    test_run_sc  = test_sc.groupby(test_run_ids.values).mean()
    train_run_sc = train_sc.groupby(train_run_ids.values).mean()

    # separation index: (train mean - test mean) / train std
    # 낮을수록 이상이므로 train_mean > test_mean → 양수 = 분리 양호
    sep = (train_run_sc.mean() - test_run_sc.mean()) / train_run_sc.std()

    print(f"\n[{name}]")
    print(f"  predicted positive rate : {pos_rate:.4f} (target ~0.322)")
    print(f"  train run scores : mean={train_run_sc.mean():.4f}  std={train_run_sc.std():.4f}")
    print(f"  test  run scores : mean={test_run_sc.mean():.4f}  std={test_run_sc.std():.4f}  "
          f"min={test_run_sc.min():.4f}")
    print(f"  separation index : {sep:.3f}")

    save_submission(pred, test_X.index, OUTPUT_DIR / f"output_{name.lower().replace(' ', '_')}.csv")
    return pred, test_run_sc, train_run_sc, sep


def main():
    np.random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    # 1단계: StandardScaler + n_clusters 탐색
    scaler = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    print("=" * 60)
    print("1단계: StandardScaler + n_clusters 탐색")
    print("=" * 60)

    results_std = {}
    best_sep = -np.inf
    best_k = None

    for k in K_VALUES:
        name = f"KMeans-k{k}-std"
        model = KMeansAnomalyDetector(n_clusters=k, random_state=RANDOM_SEED)
        model.fit(train_X)
        pred, test_sc, train_sc, sep = run_one(name, model, train_X, test_X, test_run_ids, train_run_ids)
        results_std[k] = (pred, test_sc, train_sc, sep, model)
        if sep > best_sep:
            best_sep = sep
            best_k = k

    print(f"\n→ 최적 k = {best_k} (separation index {best_sep:.3f})")

    # 2단계: 최적 k로 RobustScaler 비교
    print("\n" + "=" * 60)
    print(f"2단계: RobustScaler + n_clusters={best_k} 비교")
    print("=" * 60)

    scaler_r = fit_scaler(select_features(train_data), scaler_type="robust")
    train_X_r = scale_features(select_features(train_data), scaler_r)
    test_X_r  = scale_features(select_features(test_data),  scaler_r)

    name_r = f"KMeans-k{best_k}-robust"
    model_r = KMeansAnomalyDetector(n_clusters=best_k, random_state=RANDOM_SEED)
    model_r.fit(train_X_r)
    pred_r, test_sc_r, train_sc_r, sep_r = run_one(
        name_r, model_r, train_X_r, test_X_r, test_run_ids, train_run_ids
    )

    print(f"\n  Standard sep={results_std[best_k][3]:.3f} vs Robust sep={sep_r:.3f}")
    final_scaler = "robust" if sep_r > results_std[best_k][3] else "standard"
    print(f"  → 최종 스케일러: {final_scaler}")

    # 시각화: n_clusters 비교 히스토그램 (2×2)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"k={k}" for k in K_VALUES],
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for k, (row, col) in zip(K_VALUES, positions):
        pred, test_sc, train_sc, sep, _ = results_std[k]
        fig.add_trace(go.Histogram(
            x=test_sc.values, name=f"k={k} test", opacity=0.65, nbinsx=50,
            marker_color="steelblue",
        ), row=row, col=col)
        fig.add_trace(go.Histogram(
            x=train_sc.values, name=f"k={k} train(정상)", opacity=0.65, nbinsx=30,
            marker_color="salmon",
        ), row=row, col=col)

    fig.update_layout(
        title="KMeans n_clusters 비교 (steelblue=test, salmon=train정상 / 낮을수록 이상)",
        height=700, width=1100, barmode="overlay", showlegend=False,
    )
    out = EDA_PATH / "kmeans_grid_distribution.html"
    fig.write_html(str(out))
    print(f"\n시각화 저장: {out}")

    # LOF-A와 예측 일치도 비교
    import os
    lof_path = OUTPUT_DIR / "output_lof-a.csv"
    if lof_path.exists():
        lof_pred = pd.read_csv(lof_path)["faultNumber"].values
        print("\n[LOF-A와 run 단위 일치도 비교]")
        for k in K_VALUES:
            km_pred = results_std[k][0]
            km_run  = km_pred.reshape(-1, 960)[:, 0]
            lof_run = lof_pred.reshape(-1, 960)[:, 0]
            agree = (km_run == lof_run).sum()
            diff_km_only = ((km_run==1) & (lof_run==0)).sum()
            diff_lof_only = ((km_run==0) & (lof_run==1)).sum()
            print(f"  k={k:2d}: 일치 {agree}/740, KMeans만 이상 {diff_km_only}, LOF만 이상 {diff_lof_only}")

    print("\n=== 완료. kmeans_grid_distribution.html 확인 후 제출 모델 결정 ===")


if __name__ == "__main__":
    main()
