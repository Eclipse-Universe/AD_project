"""
run_maxpool.py — 최대 윈도우 집계 이상 탐지

현재 접근: run 전체 960 timestep 평균 → 이상이 일부 구간에 국한되면 신호 희석.
이 방법:  각 run을 W개 timestep 윈도우로 분할 → 각 윈도우를 scoring → 최대값 사용.

물리적 근거:
  TEP fault는 timestep 20에서 시작하지만 일부 fault는 초반엔 약하고 후반에 강해짐.
  또는, fault가 특정 구간에서만 통계적으로 유의미한 이상을 만들 수 있음.
  평균보다 MAX가 이런 패턴을 더 잘 포착.

윈도우 탐색: W=48(20개 윈도우), W=96(10개), W=192(5개)
집계: mean(기존), max(새), mean_of_top3(절충)

실행: cd /root/AD_project/src && python run_maxpool.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42
TEST_TS           = 960   # test run당 timestep 수

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
}

WINDOW_SIZES = [48, 96, 192, 480]  # W
AGGR_FUNCS   = ["mean", "max", "top3"]

WEIGHT_GRID = [
    (0.5, 0.3, 0.2),
    (0.45, 0.45, 0.1),
    (0.5, 0.4, 0.1),
    (0.4, 0.4, 0.2),
    (0.6, 0.3, 0.1),
]


def sep_index(tr, te):
    return float((te.mean() - tr.mean()) / (tr.std() + 1e-10))


def compare(pred_row: np.ndarray) -> str:
    pred_run = pred_row.reshape(-1, TEST_TS)[:, 0]
    parts = []
    for name, path in REF_FILES.items():
        if not Path(path).exists():
            continue
        ref   = pd.read_csv(path)["faultNumber"].values.reshape(-1, TEST_TS)[:, 0]
        agree = (pred_run == ref).sum()
        me    = ((pred_run == 1) & (ref == 0)).sum()
        ro    = ((pred_run == 0) & (ref == 1)).sum()
        parts.append(f"{name}:{agree}/740(+{me}/-{ro})")
    return "  ".join(parts)


def aggregate_window_scores(scores_2d: np.ndarray, aggr: str) -> np.ndarray:
    """scores_2d: (n_windows, n_runs). → per-run score (n_runs,)"""
    if aggr == "mean":
        return scores_2d.mean(axis=0)
    elif aggr == "max":
        return scores_2d.max(axis=0)
    elif aggr == "top3":
        n = scores_2d.shape[0]
        k = max(1, min(3, n))
        return np.sort(scores_2d, axis=0)[-k:].mean(axis=0)
    raise ValueError(f"Unknown aggr: {aggr}")


def compute_window_scores(model, X_arr, run_ids_arr, window_size: int, n_test_ts: int):
    """
    각 run을 window_size 짜리 윈도우로 분할하고, 윈도우 평균 벡터를 scoring.
    train: 500 timestep이므로 윈도우 수 다를 수 있음.
    test : 960 timestep.
    반환: (n_windows, n_runs) score matrix, run_order list
    """
    groups = {}
    for i, rid in enumerate(run_ids_arr):
        groups.setdefault(int(rid), []).append(i)
    run_order = sorted(groups.keys())

    run_windows = []  # list of (n_win_i, d) arrays
    for rid in run_order:
        rows = X_arr[np.array(groups[rid])]  # (T, d)
        T    = len(rows)
        n_win = T // window_size
        if n_win == 0:
            n_win = 1
            window_size_actual = T
        else:
            window_size_actual = window_size

        wins = []
        for w in range(n_win):
            start = w * window_size_actual
            end   = start + window_size_actual
            wins.append(rows[start:end].mean(axis=0))
        run_windows.append(np.stack(wins))  # (n_win, d)

    # Determine max windows per run and pad for matrix alignment
    n_wins_list = [rw.shape[0] for rw in run_windows]
    max_wins = max(n_wins_list)

    # Score each window for each run
    # For efficiency, batch all windows together
    all_win_vecs = []
    win_run_idx  = []  # which run each window belongs to
    win_idx_in_run = []  # which window within the run
    for run_i, rw in enumerate(run_windows):
        for win_j in range(rw.shape[0]):
            all_win_vecs.append(rw[win_j])
            win_run_idx.append(run_i)
            win_idx_in_run.append(win_j)

    all_win_vecs = np.stack(all_win_vecs)

    if hasattr(model, "score_samples"):
        raw_scores = -model.score_samples(all_win_vecs)   # GMM
    else:
        raw_scores = -model.decision_function(all_win_vecs)  # Mahal

    # Reorganize into (max_wins, n_runs) matrix
    n_runs = len(run_order)
    score_matrix = np.full((max_wins, n_runs), np.nan)
    for sc, ri, wi in zip(raw_scores, win_run_idx, win_idx_in_run):
        score_matrix[wi, ri] = sc

    # Fill NaN (runs with fewer windows) with mean of that run's windows
    for ri in range(n_runs):
        col = score_matrix[:, ri]
        valid = col[~np.isnan(col)]
        if len(valid) < max_wins:
            score_matrix[np.isnan(col), ri] = valid.mean() if len(valid) else 0.0

    return score_matrix, run_order


def main():
    print("데이터 로딩...")
    train_data    = load_train(DATA_PATH)
    test_data     = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    tr_X_df = scale_features(select_features(train_data), scaler)
    te_X_df = scale_features(select_features(test_data),  scaler)
    tr_X    = tr_X_df.values.astype(np.float64)
    te_X    = te_X_df.values.astype(np.float64)

    # 기존 run 평균 기반 GMM/Mahal (참조 스코어 — 앙상블용)
    print("GMM+Mahal (run 평균, 기존) 학습 중...")
    tr_run_means = tr_X_df.groupby(train_run_ids.values).mean().values
    te_run_vecs  = te_X_df.groupby(test_run_ids.values).mean()
    te_run_order_mean = te_run_vecs.index.tolist()
    te_run_means = te_run_vecs.values

    gmm_ref = GaussianMixture(
        n_components=5, covariance_type="tied",
        reg_covar=1e-6, n_init=10, max_iter=300, random_state=RANDOM_SEED,
    )
    gmm_ref.fit(tr_run_means)
    gmm_tr_ref = -gmm_ref.score_samples(tr_run_means)
    gmm_te_ref = -gmm_ref.score_samples(te_run_means)
    print(f"  GMM sep={sep_index(gmm_tr_ref, gmm_te_ref):.1f}")

    mah_ref = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah_ref.fit(tr_run_means)
    mah_tr_ref = -mah_ref.decision_function(tr_run_means)
    mah_te_ref = -mah_ref.decision_function(te_run_means)
    print(f"  Mahal sep={sep_index(mah_tr_ref, mah_te_ref):.1f}")

    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg

    zg_te, zg_tr = z(gmm_tr_ref, gmm_te_ref)
    zm_te, zm_tr = z(mah_tr_ref, mah_te_ref)

    best_results = []

    for W in WINDOW_SIZES:
        print(f"\n{'='*60}")
        print(f"윈도우 크기 W={W}  (test: {TEST_TS//W} windows/run, "
              f"train: {500//W} windows/run)")
        print(f"{'='*60}")

        # GMM 윈도우 모델 (run 평균 기반 GMM으로 윈도우 벡터 스코어링)
        print("  GMM 윈도우 스코어 계산 중...")
        tr_score_mat, tr_run_order_w = compute_window_scores(
            gmm_ref, tr_X, train_run_ids.values, W, 500
        )
        te_score_mat, te_run_order_w = compute_window_scores(
            gmm_ref, te_X, test_run_ids.values, W, TEST_TS
        )

        for aggr in AGGR_FUNCS:
            tr_win_sc = aggregate_window_scores(tr_score_mat, aggr)
            te_win_sc = aggregate_window_scores(te_score_mat, aggr)
            win_sep   = sep_index(tr_win_sc, te_win_sc)

            # 단독 예측 (윈도우 집계 GMM만)
            thr      = np.quantile(te_win_sc, 1 - RUN_CONTAMINATION)
            pred_run = (te_win_sc >= thr).astype(int)
            pred_row = test_run_ids.map(
                pd.Series(pred_run, index=te_run_order_w)
            ).values
            cmp_alone = compare(pred_row)

            # 앙상블: 기존 GMM + Mahal + 윈도우GMM
            zw_te_raw, zw_tr_raw = z(tr_win_sc, te_win_sc)
            best_ens_sep = -np.inf
            best_ens_row = None
            best_ens_w   = None
            best_ens_cmp = None

            for wg, wm, ww in WEIGHT_GRID:
                ens_te = wg * zg_te + wm * zm_te + ww * zw_te_raw
                ens_tr = wg * zg_tr + wm * zm_tr + ww * zw_tr_raw
                sep_ens = sep_index(ens_tr, ens_te)
                thr2    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
                pr2     = (ens_te >= thr2).astype(int)
                row2    = test_run_ids.map(
                    pd.Series(pr2, index=te_run_order_mean)
                ).values
                if sep_ens > best_ens_sep:
                    best_ens_sep = sep_ens
                    best_ens_row = row2
                    best_ens_w   = (wg, wm, ww)
                    best_ens_cmp = compare(row2)

            tag = f"maxpool_W{W}_{aggr}"
            save_submission(best_ens_row, te_X_df.index,
                            OUTPUT_DIR / f"output_{tag}.csv")

            print(f"  W={W:4d} {aggr:6s}: win_sep={win_sep:9.1f}  "
                  f"단독: {cmp_alone}  |  "
                  f"앙상블(w={best_ens_w}): {best_ens_cmp}")

            best_results.append({
                "W": W, "aggr": aggr, "win_sep": win_sep,
                "alone": cmp_alone, "ens_sep": best_ens_sep,
                "ens_w": best_ens_w, "ens_cmp": best_ens_cmp,
            })

    print("\n" + "="*70)
    print("요약 (앙상블 sep 내림차순)")
    print("="*70)
    best_results.sort(key=lambda r: r["ens_sep"], reverse=True)
    for r in best_results[:10]:
        print(f"  W={r['W']:4d} {r['aggr']:6s}  "
              f"win_sep={r['win_sep']:9.1f}  "
              f"ens_sep={r['ens_sep']:9.1f}  {r['ens_cmp']}")


if __name__ == "__main__":
    main()
