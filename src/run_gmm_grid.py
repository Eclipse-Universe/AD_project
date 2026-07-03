"""GMM (Gaussian Mixture Model) 이상 탐지 — run-level 방식, n_components 탐색.

전략 (run-level GMM):
  row-level(250K행) GMM은 full covariance 추정 비용이 너무 크다.
  대신 훈련 run 500개의 평균 벡터(500×52)로 GMM을 학습한다.
  - 학습 데이터: 500행 → 즉각 수렴
  - 이상 점수: 740개 test run 평균 벡터에 score_samples 직접 적용
  - 수치 안정성: 500/n_components ≥ 25 (52×52 Σ_k 추정에 충분)

이상 점수 방향:
  score_samples(x) = log p(x): 낮을수록(확률 낮을수록) 이상.

이론적 근거:
  Bishop (2006) PRML Ch.9 — GMM EM 추정 및 로그우도 밀도 추정
  Markou & Singh (2003) Pattern Recognition Letters — GMM novelty detection 타당성

실행: cd src && python run_gmm_grid.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
EDA_PATH = Path("/root/AD_project/eda")

RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42
N_COMPONENTS = [5, 10, 20]


def build_run_vectors(X: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    return X.groupby(run_ids.values).mean()


def separation_index(train_sc: np.ndarray, test_sc: np.ndarray) -> float:
    return (train_sc.mean() - test_sc.mean()) / train_sc.std()


def main():
    np.random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    train_vecs = build_run_vectors(train_X, train_run_ids)   # (500, 52)
    test_vecs  = build_run_vectors(test_X,  test_run_ids)    # (740, 52)

    print("=" * 65)
    print("GMM run-level 이상 탐지 — n_components 탐색")
    print(f"  훈련 run: {len(train_vecs)}  테스트 run: {len(test_vecs)}  피처: {train_X.shape[1]}")
    print("=" * 65)

    results = {}
    best_sep, best_k = -np.inf, None

    for k in N_COMPONENTS:
        print(f"\n[학습 중] n_components={k} ...")
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_SEED,
            max_iter=300,
            n_init=5,
        )
        gmm.fit(train_vecs.values)

        train_scores = gmm.score_samples(train_vecs.values)
        test_scores  = gmm.score_samples(test_vecs.values)

        sep = separation_index(train_scores, test_scores)

        threshold     = np.quantile(test_scores, RUN_CONTAMINATION)
        anomalous_runs = test_vecs.index[test_scores <= threshold]
        pred_run      = pd.Series(0, index=test_vecs.index)
        pred_run[anomalous_runs] = 1
        pred_rows = test_run_ids.map(pred_run).values
        pos_rate  = pred_rows.mean()

        print(f"[GMM-k{k}]")
        print(f"  predicted positive rate : {pos_rate:.4f} (target ~0.322)")
        print(f"  train run : mean={train_scores.mean():.4f}  std={train_scores.std():.4f}")
        print(f"  test  run : mean={test_scores.mean():.4f}  std={test_scores.std():.4f}  "
              f"min={test_scores.min():.4f}")
        print(f"  separation index : {sep:.3f}")

        save_submission(pred_rows, test_X.index, OUTPUT_DIR / f"output_gmm-k{k}.csv")
        results[k] = (pred_rows, test_scores, train_scores, sep)

        if sep > best_sep:
            best_sep, best_k = sep, k

    print(f"\n→ 최적 n_components = {best_k} (separation index {best_sep:.3f})")

    print("\n" + "=" * 65)
    print("LOF-A(Exp 15) / KMeans-Mahal(Exp 17)과 run 단위 일치도")
    print("=" * 65)

    ref_files = {
        "LOF-A(Exp15)":        OUTPUT_DIR / "output_exp15(LOF).csv",
        "KMeans-Mahal(Exp17)": OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv",
    }
    for ref_name, ref_path in ref_files.items():
        if not ref_path.exists():
            print(f"  {ref_name}: 파일 없음")
            continue
        ref_run = pd.read_csv(ref_path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        print(f"\n  기준: {ref_name}")
        for k in N_COMPONENTS:
            gmm_run = np.array(results[k][0]).reshape(-1, 960)[:, 0]
            agree    = (gmm_run == ref_run).sum()
            gmm_only = ((gmm_run == 1) & (ref_run == 0)).sum()
            ref_only = ((gmm_run == 0) & (ref_run == 1)).sum()
            print(f"    GMM-k{k}: 일치 {agree}/740 | "
                  f"GMM만 이상 {gmm_only} | {ref_name.split('(')[0]}만 이상 {ref_only}")

    fig = make_subplots(rows=1, cols=len(N_COMPONENTS),
                        subplot_titles=[f"GMM k={k}" for k in N_COMPONENTS])
    for idx, k in enumerate(N_COMPONENTS):
        _, test_sc, train_sc, _ = results[k]
        fig.add_trace(go.Histogram(x=test_sc, name=f"k={k} test",
                                   opacity=0.65, nbinsx=50, marker_color="steelblue"),
                      row=1, col=idx+1)
        fig.add_trace(go.Histogram(x=train_sc, name=f"k={k} train",
                                   opacity=0.65, nbinsx=30, marker_color="salmon"),
                      row=1, col=idx+1)
    fig.update_layout(
        title="GMM run-level 점수 분포 (steelblue=test, salmon=train정상 / 낮을수록 이상)",
        height=450, width=1200, barmode="overlay", showlegend=False,
    )
    fig.write_html(str(EDA_PATH / "gmm_distribution.html"))

    best_dst = OUTPUT_DIR / f"output_exp20(GMM).csv"
    shutil.copy(OUTPUT_DIR / f"output_gmm-k{best_k}.csv", best_dst)
    print(f"\n제출 파일: {best_dst}")


if __name__ == "__main__":
    main()
