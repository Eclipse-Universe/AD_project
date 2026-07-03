"""KNN run-level 이상 탐지 — k 탐색 및 LOF/KMeans-Mahal과 비교.

전략: run-level 평균 벡터 기반 KNN
  - row-level KNN이 아닌 run 평균 52차원 벡터를 사용한다.
  - 이유 1) 계산: train 250K행 × test 711K행 KNN은 메모리 폭발 위험.
  - 이유 2) 의미: LOF/KMeans도 row 점수 → run 집계로 판정하므로
             KNN도 동일 레벨(run)에서 비교해야 공정한 비교가 된다.
  - 이유 3) 물리적 타당성: run 하나의 500 timestep 평균이 해당 run의
             "정상 운전점"을 대표한다 — 일시 변동보다 구조적 이탈이 중요.

k 탐색: k∈{5, 10, 20, 50} — 훈련 run 500개 대비 1~10% 범위.
비교: Exp 15(LOF), Exp 17(KMeans-Mahal)과 run 단위 일치도.

실행: cd src && python run_knn_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

from data_loader import load_test, load_train
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42
K_VALUES = [5, 10, 20, 50]


def build_run_vectors(X: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    """각 run의 timestep 평균 벡터를 반환한다. index = simulationRun."""
    return X.groupby(run_ids.values).mean()


def knn_run_level(train_run_vecs: pd.DataFrame, test_run_vecs: pd.DataFrame, k: int):
    """train run 벡터에 KNN 학습 → test run 벡터의 k-이웃 평균 거리 반환.

    거리가 클수록 이상 — 음수화해 decision_function 방향 통일.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
    nn.fit(train_run_vecs.values)
    dists, _ = nn.kneighbors(test_run_vecs.values)       # (740, k)
    mean_dists = dists.mean(axis=1)                       # (740,)
    return -mean_dists                                     # 낮을수록 이상


def threshold_and_predict(run_scores: np.ndarray, run_ids: pd.Series,
                           test_run_order: pd.Index, test_run_ids_full: pd.Series) -> np.ndarray:
    """run_scores → 각 row에 run 예측 값 확장."""
    threshold = np.quantile(run_scores, RUN_CONTAMINATION)
    anomalous = test_run_order[run_scores <= threshold]
    return test_run_ids_full.isin(anomalous).astype(int).values


def separation_index(train_scores: np.ndarray, test_scores: np.ndarray) -> float:
    return (train_scores.mean() - test_scores.mean()) / train_scores.std()


def main():
    np.random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler   = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X  = scale_features(select_features(train_data), scaler)
    test_X   = scale_features(select_features(test_data),  scaler)

    train_run_vecs = build_run_vectors(train_X, train_run_ids)   # (500, 52)
    test_run_vecs  = build_run_vectors(test_X,  test_run_ids)    # (740, 52)

    print("=" * 65)
    print("KNN run-level 이상 탐지 — k 탐색")
    print(f"  훈련 run 수: {len(train_run_vecs)}  테스트 run 수: {len(test_run_vecs)}")
    print(f"  피처 수: {train_X.shape[1]}")
    print("=" * 65)

    # train run의 KNN 자기 거리 (leave-one-out 근사: k+1 이웃 → 자기 제외)
    nn_self = NearestNeighbors(n_neighbors=max(K_VALUES) + 1, algorithm="ball_tree")
    nn_self.fit(train_run_vecs.values)
    train_self_dists, _ = nn_self.kneighbors(train_run_vecs.values)

    results = {}
    best_sep, best_k = -np.inf, None

    for k in K_VALUES:
        # train 자기 거리: 첫 이웃(자기 자신=0)을 제외하고 k개 이웃 평균
        train_k_dists = train_self_dists[:, 1:k+1].mean(axis=1)
        train_scores  = -train_k_dists                      # 낮을수록 이상

        test_scores = knn_run_level(train_run_vecs, test_run_vecs, k)

        sep = separation_index(train_scores, test_scores)
        pos_rate = (test_scores <= np.quantile(test_scores, RUN_CONTAMINATION)).mean()

        print(f"\n[KNN-k{k}]")
        print(f"  predicted positive rate : {pos_rate:.4f} (target ~0.320)")
        print(f"  train run : mean={train_scores.mean():.4f}  std={train_scores.std():.4f}")
        print(f"  test  run : mean={test_scores.mean():.4f}  std={test_scores.std():.4f}  "
              f"min={test_scores.min():.4f}")
        print(f"  separation index : {sep:.3f}")

        pred = threshold_and_predict(test_scores, test_run_ids, test_run_vecs.index, test_run_ids)
        results[k] = (pred, test_scores, train_scores, sep)

        save_submission(pred, test_data.index,
                        OUTPUT_DIR / f"output_knn-k{k}.csv")

        if sep > best_sep:
            best_sep, best_k = sep, k

    print(f"\n→ 최적 k = {best_k} (separation index {best_sep:.3f})")

    # 일치도 비교 — LOF-A(Exp15), KMeans-Mahal(Exp17)
    print("\n" + "=" * 65)
    print("LOF-A(Exp 15) / KMeans-Mahal(Exp 17)과 run 단위 일치도")
    print("=" * 65)

    ref_files = {
        "LOF-A(Exp15)":      OUTPUT_DIR / "output_exp15(LOF).csv",
        "KMeans-Mahal(Exp17)": OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv",
    }
    for ref_name, ref_path in ref_files.items():
        if not ref_path.exists():
            print(f"  {ref_name}: 파일 없음")
            continue
        ref_pred = pd.read_csv(ref_path)["faultNumber"].values
        ref_run  = ref_pred.reshape(-1, 960)[:, 0]
        print(f"\n  기준: {ref_name}")
        for k in K_VALUES:
            knn_run = np.array(results[k][0]).reshape(-1, 960)[:, 0]
            agree    = (knn_run == ref_run).sum()
            knn_only = ((knn_run == 1) & (ref_run == 0)).sum()
            ref_only = ((knn_run == 0) & (ref_run == 1)).sum()
            print(f"    KNN-k{k}: 일치 {agree}/740 | "
                  f"KNN만 이상 {knn_only} | {ref_name.split('(')[0]}만 이상 {ref_only}")

    # Rename best k output to Exp 18
    best_src = OUTPUT_DIR / f"output_knn-k{best_k}.csv"
    best_dst = OUTPUT_DIR / f"output_exp18(KNN).csv"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        print(f"\n제출 파일: {best_dst}")

    # 시각화: k별 run 점수 분포
    fig = make_subplots(rows=1, cols=len(K_VALUES),
                        subplot_titles=[f"KNN k={k}" for k in K_VALUES])
    for idx, k in enumerate(K_VALUES):
        _, test_sc, train_sc, _ = results[k]
        fig.add_trace(go.Histogram(x=test_sc, name=f"k={k} test",
                                   opacity=0.65, nbinsx=50, marker_color="steelblue"),
                      row=1, col=idx+1)
        fig.add_trace(go.Histogram(x=train_sc, name=f"k={k} train",
                                   opacity=0.65, nbinsx=30, marker_color="salmon"),
                      row=1, col=idx+1)
    fig.update_layout(
        title="KNN run-level 점수 분포 (steelblue=test, salmon=train정상 / 낮을수록 이상)",
        height=450, width=1200, barmode="overlay", showlegend=False,
    )
    out = EDA_PATH / "knn_distribution.html"
    fig.write_html(str(out))
    print(f"시각화: {out}")


if __name__ == "__main__":
    main()
