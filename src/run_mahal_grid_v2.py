"""KMeans-Mahal 재탐색 — n_clusters 확장 + Local Mahal 비교.

현재 상태:
  Global Mahal k=50 (Exp17, F1=0.9277)
  k=100 탐색 → sep 소폭 개선이나 이진 예측 동일 → 미제출

두 가지 탐색:
  A. Global Mahal: n_clusters {50, 100, 150, 200}
     단일 Σ⁻¹, 클러스터 수만 변경

  B. Local Mahal: n_clusters {50, 100, 150}
     클러스터별 Σ_k⁻¹ — TEP 운전조건별 공분산 분리
     이론: TEP 데이터는 여러 운전 조건 혼재 → Global Σ⁻¹는 이를 평균화해 근사
          Local Σ_k는 각 운전 구간의 센서 상관 구조를 개별 학습

실행: cd src && python run_mahal_grid_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector, LocalMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

GLOBAL_K_GRID = [50, 100, 150, 200]
LOCAL_K_GRID  = [50, 100, 150]

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
    "Exp17": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
}


def sep_index(tr_scores: np.ndarray, te_scores: np.ndarray) -> float:
    return (te_scores.mean() - tr_scores.mean()) / tr_scores.std()


def compare(pred_row: np.ndarray) -> str:
    """pred_row: row-level 예측 (원본 test 데이터 순서). reshape로 run-level 비교."""
    pred_run = pred_row.reshape(-1, 960)[:, 0]
    parts = []
    for name, path in REF_FILES.items():
        if not Path(path).exists():
            continue
        ref   = pd.read_csv(path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree = (pred_run == ref).sum()
        me    = ((pred_run == 1) & (ref == 0)).sum()
        ro    = ((pred_run == 0) & (ref == 1)).sum()
        parts.append(f"{name}:{agree}/740(+{me}/-{ro})")
    return "  ".join(parts)


def score_run(raw_scores: np.ndarray, ids: pd.Series, run_idx) -> np.ndarray:
    """row-level 점수 → run-level 평균 점수."""
    return pd.Series(raw_scores, index=ids.index).groupby(ids.values).mean().values


def predict_runs(run_scores: np.ndarray, run_idx) -> np.ndarray:
    thr      = np.quantile(run_scores, 1 - RUN_CONTAMINATION)
    pred_run = (run_scores >= thr).astype(int)
    return pred_run


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler   = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X  = scale_features(select_features(train_data), scaler)
    test_X   = scale_features(select_features(test_data),  scaler)
    run_idx  = test_X.groupby(test_run_ids.values).mean().index

    print("=" * 72)
    print(f"Mahal 재탐색: train={train_X.shape}  test={test_X.shape}")
    print("=" * 72)

    # ── A. Global Mahal (기존 방식, n_clusters 확장)
    print("\n" + "─" * 72)
    print("A. Global Mahal (단일 Σ⁻¹)")
    print("─" * 72)
    print(f"  {'k':>5}  {'sep':>10}  {'pos%':>6}  {'Exp25':>14}  {'Exp17':>14}")

    global_results = []
    for k in GLOBAL_K_GRID:
        print(f"  [k={k} 학습 중...]", end="", flush=True)
        det = KMeansMahalanobisDetector(n_clusters=k, random_state=RANDOM_SEED)
        det.fit(train_X)

        tr_raw = -det.decision_function(train_X)
        te_raw = -det.decision_function(test_X)
        tr_sc  = score_run(tr_raw, train_run_ids, None)
        te_sc  = pd.Series(te_raw, index=test_X.index).groupby(test_run_ids.values).mean().values
        sep    = sep_index(tr_sc, te_sc)

        pred_run = predict_runs(te_sc, run_idx)
        pred_row = test_run_ids.map(
            pd.Series(pred_run, index=run_idx)
        ).values
        pos = pred_run.mean()

        cmp = compare(pred_row)
        print(f"\r  {k:>5}  {sep:>10.1f}  {pos:>6.4f}  {cmp}")

        tag = f"mahal_global_k{k}"
        save_submission(pred_row, test_X.index, OUTPUT_DIR / f"output_{tag}.csv")
        global_results.append(dict(type="global", k=k, sep=sep, pos=pos))

    # ── B. Local Mahal (클러스터별 Σ_k⁻¹)
    print("\n" + "─" * 72)
    print("B. Local Mahal (클러스터별 Σ_k⁻¹ — 운전구간별 공분산 분리)")
    print("─" * 72)
    print(f"  {'k':>5}  {'sep':>10}  {'pos%':>6}  {'Exp25':>14}  {'Exp17':>14}")

    local_results = []
    for k in LOCAL_K_GRID:
        print(f"  [k={k} 학습 중 (Local Mahal)...]", end="", flush=True)
        det = LocalMahalanobisDetector(n_clusters=k, reg=1e-6, random_state=RANDOM_SEED)
        det.fit(train_X)

        tr_raw = -det.decision_function(train_X)
        te_raw = -det.decision_function(test_X)
        tr_sc  = score_run(tr_raw, train_run_ids, None)
        te_sc  = pd.Series(te_raw, index=test_X.index).groupby(test_run_ids.values).mean().values
        sep    = sep_index(tr_sc, te_sc)

        pred_run = predict_runs(te_sc, run_idx)
        pred_row = test_run_ids.map(
            pd.Series(pred_run, index=run_idx)
        ).values
        pos = pred_run.mean()

        cmp = compare(pred_row)
        print(f"\r  {k:>5}  {sep:>10.1f}  {pos:>6.4f}  {cmp}")

        tag = f"mahal_local_k{k}"
        save_submission(pred_row, test_X.index, OUTPUT_DIR / f"output_{tag}.csv")
        local_results.append(dict(type="local", k=k, sep=sep, pos=pos))

    # 요약
    all_results = global_results + local_results
    df = pd.DataFrame(all_results).sort_values("sep", ascending=False)
    print("\n" + "=" * 72)
    print("요약 — sep 내림차순")
    print("=" * 72)
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\n→ sep 최적: type={best.type}  k={int(best.k)}  sep={best.sep:.1f}")


if __name__ == "__main__":
    main()
