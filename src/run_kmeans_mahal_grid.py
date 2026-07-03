"""KMeans + 마할라노비스 거리 — n_clusters 탐색 및 유클리드 KMeans와 비교.

목적:
  유클리드 거리(Exp 16)의 한계(피처 독립 가정, 상관 구조 미반영)를 마할라노비스 거리로 극복.
  k∈{20, 30, 50}으로 탐색하고 separation index + LOF/KMeans_Euclidean 일치도로 비교.

핵심 가설:
  - xmv_7↔xmeas_12처럼 정상 조건에서 강하게 상관된 피처 쌍이 이상 상황에서 관계가 깨진다.
  - 유클리드 거리는 이 신호를 포착하지 못하지만, 마할라노비스 거리는 Σ⁻¹에 상관 구조가
    인코딩되어 있어 관계가 깨진 run에 더 큰 거리 값을 부여한다.

실행: cd src && python run_kmeans_mahal_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_loader import load_test, load_train
from infer import predict_labels_by_run, save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42
K_VALUES = [20, 30, 50]


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

    sep = (train_run_sc.mean() - test_run_sc.mean()) / train_run_sc.std()

    print(f"\n[{name}]")
    print(f"  predicted positive rate : {pos_rate:.4f} (target ~0.322)")
    print(f"  train run : mean={train_run_sc.mean():.4f}  std={train_run_sc.std():.4f}")
    print(f"  test  run : mean={test_run_sc.mean():.4f}  std={test_run_sc.std():.4f}  "
          f"min={test_run_sc.min():.4f}")
    print(f"  separation index : {sep:.3f}")

    save_submission(pred, test_X.index, OUTPUT_DIR / f"output_{name.lower().replace(' ','_')}.csv")
    return pred, test_run_sc, train_run_sc, sep


def main():
    np.random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler   = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X  = scale_features(select_features(train_data), scaler)
    test_X   = scale_features(select_features(test_data),  scaler)

    print("=" * 65)
    print("KMeans + 마할라노비스 거리 — n_clusters 탐색")
    print("=" * 65)

    results = {}
    best_sep, best_k = -np.inf, None

    for k in K_VALUES:
        name = f"KMeans-Mahal-k{k}"
        print(f"\n[학습 중] k={k} ...")
        model = KMeansMahalanobisDetector(n_clusters=k, random_state=RANDOM_SEED)
        model.fit(train_X)
        pred, test_sc, train_sc, sep = run_one(
            name, model, train_X, test_X, test_run_ids, train_run_ids
        )
        results[k] = (pred, test_sc, train_sc, sep)
        if sep > best_sep:
            best_sep, best_k = sep, k

    print(f"\n→ 최적 k = {best_k} (separation index {best_sep:.3f})")

    # 비교: 유클리드 KMeans-k50(Exp 16), LOF-A(Exp 15) 대비 일치도
    print("\n" + "=" * 65)
    print("LOF-A(Exp 15) / KMeans-Euclid-k50(Exp 16)와 run 단위 일치도")
    print("=" * 65)

    ref_files = {
        "LOF-A(Exp15)":       OUTPUT_DIR / "output_exp15(LOF).csv",
        "KMeans-Euclid(Exp16)": OUTPUT_DIR / "output_exp16(KMeans).csv",
    }
    for ref_name, ref_path in ref_files.items():
        if not ref_path.exists():
            print(f"  {ref_name}: 파일 없음 ({ref_path})")
            continue
        ref_pred = pd.read_csv(ref_path)["faultNumber"].values
        ref_run  = ref_pred.reshape(-1, 960)[:, 0]
        print(f"\n  기준: {ref_name}")
        for k in K_VALUES:
            km_pred = results[k][0]
            km_run  = np.array(km_pred).reshape(-1, 960)[:, 0]
            agree     = (km_run == ref_run).sum()
            mahal_only = ((km_run == 1) & (ref_run == 0)).sum()
            ref_only   = ((km_run == 0) & (ref_run == 1)).sum()
            print(f"    Mahal-k{k}: 일치 {agree}/740 | "
                  f"Mahal만 이상 {mahal_only} | {ref_name.split('(')[0]}만 이상 {ref_only}")

    # 시각화: k별 run-level 점수 분포
    fig = make_subplots(rows=1, cols=len(K_VALUES),
                        subplot_titles=[f"Mahal k={k}" for k in K_VALUES])
    for idx, k in enumerate(K_VALUES):
        _, test_sc, train_sc, _ = results[k]
        fig.add_trace(go.Histogram(x=test_sc.values, name=f"k={k} test",
                                   opacity=0.65, nbinsx=50, marker_color="steelblue"),
                      row=1, col=idx+1)
        fig.add_trace(go.Histogram(x=train_sc.values, name=f"k={k} train",
                                   opacity=0.65, nbinsx=30, marker_color="salmon"),
                      row=1, col=idx+1)
    fig.update_layout(
        title="KMeans-Mahalanobis run-level 점수 분포 (steelblue=test, salmon=train정상 / 낮을수록 이상)",
        height=450, width=1200, barmode="overlay", showlegend=False,
    )
    out = EDA_PATH / "kmeans_mahal_distribution.html"
    fig.write_html(str(out))
    print(f"\n시각화: {out}")
    print(f"제출 파일: output_kmeans-mahal-k{best_k}.csv → 확인 후 output_exp17(KMeans-Mahal).csv로 rename")


if __name__ == "__main__":
    main()
