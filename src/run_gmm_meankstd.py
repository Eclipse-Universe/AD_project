"""
run_gmm_meankstd.py — mean+std 피처에서 GMM k 최적화

run_feature_expand.py 결과에서 mean+std가 유망할 경우 GMM k={2,3,4,5,6,8}을 재탐색.
실행: cd /root/AD_project/src && python run_gmm_meankstd.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
}

K_GRID      = [2, 3, 4, 5, 6, 8]
WEIGHT_GRID = [
    (0.5,  0.3,  0.2),
    (0.45, 0.45, 0.1),
    (0.5,  0.4,  0.1),
    (0.4,  0.4,  0.2),
    (0.6,  0.3,  0.1),
]


def compute_run_mean_std(X_arr, run_ids_arr):
    groups = {}
    for i, rid in enumerate(run_ids_arr):
        groups.setdefault(int(rid), []).append(i)
    run_order = sorted(groups.keys())
    n, d = len(run_order), X_arr.shape[1]
    means = np.zeros((n, d))
    stds  = np.zeros((n, d))
    for i, rid in enumerate(run_order):
        rows = X_arr[np.array(groups[rid])]
        means[i] = rows.mean(axis=0)
        stds[i]  = rows.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(d)
    return run_order, np.hstack([means, stds])


def sep_index(tr, te):
    return float((te.mean() - tr.mean()) / (tr.std() + 1e-10))


def compare(pred_row):
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


def main():
    print("데이터 로딩...")
    train_data    = load_train(DATA_PATH)
    test_data     = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    tr_X    = scale_features(select_features(train_data), scaler).values.astype(np.float64)
    te_X_df = scale_features(select_features(test_data),  scaler)
    te_X    = te_X_df.values.astype(np.float64)

    print("mean+std 피처 구성 중...")
    tr_run_order, tr_feats_raw = compute_run_mean_std(tr_X, train_run_ids.values)
    te_run_order, te_feats_raw = compute_run_mean_std(te_X, test_run_ids.values)

    rl_scaler = StandardScaler()
    tr_feats  = rl_scaler.fit_transform(tr_feats_raw)
    te_feats  = rl_scaler.transform(te_feats_raw)

    # Mahal + LOF는 k와 무관 → 한 번만 학습
    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_feats)
    mah_tr = -mah.decision_function(tr_feats)
    mah_te = -mah.decision_function(te_feats)

    lof = LocalOutlierFactor(novelty=True, n_neighbors=10)
    lof.fit(tr_feats)
    lof_tr = -lof.decision_function(tr_feats)
    lof_te = -lof.decision_function(te_feats)

    print(f"Mahal sep={sep_index(mah_tr,mah_te):.1f}  LOF sep={sep_index(lof_tr,lof_te):.1f}\n")

    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg

    zm_te, zm_tr = z(mah_tr, mah_te)
    zl_te, zl_tr = z(lof_tr, lof_te)

    results = []
    print(f"{'k':>3}  {'GMM_sep':>9}  {'w':>18}  {'ens_sep':>9}  결과")
    print("─" * 75)

    for k in K_GRID:
        gmm = GaussianMixture(
            n_components=k, covariance_type="tied",
            reg_covar=1e-4, n_init=5, max_iter=300, random_state=RANDOM_SEED,
        )
        gmm.fit(tr_feats)
        gmm_tr = -gmm.score_samples(tr_feats)
        gmm_te = -gmm.score_samples(te_feats)
        gsep   = sep_index(gmm_tr, gmm_te)
        zg_te, zg_tr = z(gmm_tr, gmm_te)

        best_sep, best_row, best_w, best_cmp = -np.inf, None, None, None
        for wg, wm, wl in WEIGHT_GRID:
            ens_te = wg * zg_te + wm * zm_te + wl * zl_te
            ens_tr = wg * zg_tr + wm * zm_tr + wl * zl_tr
            sep    = sep_index(ens_tr, ens_te)

            thr      = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
            pred_run = (ens_te >= thr).astype(int)
            pred_row = test_run_ids.map(
                pd.Series(pred_run, index=te_run_order)
            ).values
            if sep > best_sep:
                best_sep = sep
                best_row = pred_row
                best_w   = (wg, wm, wl)
                best_cmp = compare(pred_row)

        print(f"{k:>3}  {gsep:>9.1f}  {str(best_w):>18s}  {best_sep:>9.1f}  {best_cmp}")
        tag = f"fexp_std_k{k}"
        save_submission(best_row, te_X_df.index,
                        OUTPUT_DIR / f"output_{tag}.csv")
        results.append((k, gsep, best_sep, best_w, best_cmp))

    print("\n완료 — output_fexp_std_k*.csv 저장됨")


if __name__ == "__main__":
    main()
