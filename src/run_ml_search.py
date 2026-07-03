"""ML 모델 비교 실험 — 로컬 점수 분포 확인 후 최적 모델 1개만 제출.

LOF → KMeans → KNN 순서로 실행하고 예측 비율과 run-level 점수 분포를 출력한다.
제출 전에 어느 모델이 정상/이상 run을 가장 잘 분리하는지 확인하는 것이 목적이다.

실행: cd src && python run_ml_search.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from data_loader import load_test, load_train
from infer import predict_labels_by_run, save_submission
from model import KMeansAnomalyDetector, KNNAnomalyDetector, train_lof
from preprocess import fit_scaler, scale_features, select_features

RANDOM_SEED = 42
DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
ESTIMATED_TRUE_RATE = 0.322


def run_one(name, model, train_X, test_X, test_run_ids):
    pred = predict_labels_by_run(
        model, test_X, test_run_ids,
        run_contamination=RUN_CONTAMINATION,
        agg="mean",
    )
    counts = pd.Series(pred).value_counts().sort_index()
    rate = counts.get(1, 0) / len(pred)
    print(f"  [{name}] predicted positive rate: {rate:.4f} (target: {ESTIMATED_TRUE_RATE:.4f})")
    print(f"          label counts: {counts.to_dict()}")

    # run-level 점수 분포 계산 (시각화용)
    scores = pd.Series(model.decision_function(test_X), index=test_X.index)
    run_scores = scores.groupby(test_run_ids.values).mean()
    return pred, run_scores


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 전처리 (StandardScaler — 거리 기반 모델에 필수)
    train_data = load_train(DATA_PATH)
    train_X_raw = select_features(train_data)
    scaler = fit_scaler(train_X_raw, scaler_type="standard")
    train_X = scale_features(train_X_raw, scaler)

    test_data = load_test(DATA_PATH)
    test_X_raw = select_features(test_data)
    test_X = scale_features(test_X_raw, scaler)
    test_run_ids = test_data["simulationRun"]

    print("=" * 55)
    print("LOF (n_neighbors=20)")
    print("=" * 55)
    lof = train_lof(train_X, n_neighbors=20)
    lof_pred, lof_scores = run_one("LOF", lof, train_X, test_X, test_run_ids)
    save_submission(lof_pred, test_X.index, OUTPUT_DIR / "output_exp_lof.csv")

    print("\n" + "=" * 55)
    print("KMeans (n_clusters=20)")
    print("=" * 55)
    kmeans = KMeansAnomalyDetector(n_clusters=20, random_state=RANDOM_SEED)
    kmeans.fit(train_X)
    kmeans_pred, kmeans_scores = run_one("KMeans", kmeans, train_X, test_X, test_run_ids)
    save_submission(kmeans_pred, test_X.index, OUTPUT_DIR / "output_exp_kmeans.csv")

    print("\n" + "=" * 55)
    print("KNN (n_neighbors=10)")
    print("=" * 55)
    knn = KNNAnomalyDetector(n_neighbors=10)
    knn.fit(train_X)
    knn_pred, knn_scores = run_one("KNN", knn, train_X, test_X, test_run_ids)
    save_submission(knn_pred, test_X.index, OUTPUT_DIR / "output_exp_knn.csv")

    # run-level 점수 분포 시각화 — 세 모델 비교
    # 점수가 낮을수록 이상. 정상/이상 run이 잘 분리되면 히스토그램에 두 봉우리가 보인다.
    fig = go.Figure()
    for name, scores in [("LOF", lof_scores), ("KMeans", kmeans_scores), ("KNN", knn_scores)]:
        fig.add_trace(go.Histogram(
            x=scores.values, name=name, opacity=0.6, nbinsx=60,
        ))
    fig.update_layout(
        title="Run-level 이상 점수 분포 비교 (낮을수록 이상 — 두 봉우리면 분리 양호)",
        xaxis_title="mean decision score per run",
        barmode="overlay",
        height=500, width=900,
    )
    fig.write_html(str(EDA_PATH / "ml_score_distribution.html"))
    print("\n저장: eda/ml_score_distribution.html")

    # train run-level 점수도 함께 계산 (정상 기준선)
    train_run_ids = train_data["simulationRun"]
    print("\n[Train run 점수 — 정상 기준선]")
    for name, model in [("LOF", lof), ("KMeans", kmeans), ("KNN", knn)]:
        tr_scores = pd.Series(model.decision_function(train_X), index=train_X.index)
        tr_run = tr_scores.groupby(train_run_ids.values).mean()
        print(f"  {name}: mean={tr_run.mean():.4f}, std={tr_run.std():.4f}, "
              f"min={tr_run.min():.4f}, max={tr_run.max():.4f}")

    print("\n=== 완료. eda/ml_score_distribution.html 확인 후 제출 모델 결정 ===")


if __name__ == "__main__":
    main()
