"""
run_feature_expand.py — run-level 통계 피처 확장 체계적 탐색

현재 F1=0.9623 ceiling: run 평균 벡터 52차원의 정보 고갈.
탐색 가설:
  std    : fault → 변동성 증가. 평균이 정상이어도 std 이탈 가능.
  diff_q : last_quarter - first_quarter. fault가 늦게 발현될수록 diff 큼.
  p10/p90: 분위수로 분포 형태 포착. 비대칭 fault 탐지.
  slope  : 선형 드리프트 포착.

실행: cd /root/AD_project/src && python run_feature_expand.py
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

# Feature sets: key → list of stat names
FEATURE_SETS = {
    "baseline":          ["mean"],
    "mean+std":          ["mean", "std"],
    "mean+diff_q":       ["mean", "diff_q"],
    "mean+p10p90":       ["mean", "p10", "p90"],
    "mean+std+diff_q":   ["mean", "std", "diff_q"],
    "mean+std+slope":    ["mean", "std", "slope"],
    "all_stats":         ["mean", "std", "slope", "diff_q", "p10", "p90"],
}

WEIGHT_GRID = [
    (0.5,  0.3,  0.2),
    (0.45, 0.45, 0.1),
    (0.5,  0.4,  0.1),
    (0.4,  0.4,  0.2),
    (0.6,  0.3,  0.1),
    (0.4,  0.5,  0.1),
]


# ─── feature extraction ───────────────────────────────────────────────────────

def compute_run_stats(X_arr: np.ndarray, run_ids_arr: np.ndarray,
                      quarter_frac: float = 0.25):
    """Vectorized per-run statistics. Returns (run_order, stats_dict)."""
    groups: dict = {}
    for i, rid in enumerate(run_ids_arr):
        groups.setdefault(int(rid), []).append(i)

    run_order = sorted(groups.keys())
    n, d = len(run_order), X_arr.shape[1]

    arrs = {k: np.zeros((n, d), dtype=np.float64)
            for k in ("mean", "std", "slope", "p10", "p90", "diff_q")}

    for i, rid in enumerate(run_order):
        idx  = np.array(groups[rid], dtype=int)
        rows = X_arr[idx]          # (T, d)
        T    = len(rows)
        q    = max(1, int(T * quarter_frac))

        arrs["mean"][i]  = rows.mean(axis=0)
        arrs["std"][i]   = rows.std(axis=0, ddof=1) if T > 1 else np.zeros(d)
        arrs["p10"][i]   = np.percentile(rows, 10, axis=0)
        arrs["p90"][i]   = np.percentile(rows, 90, axis=0)

        t    = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
        norm = float((t * t).sum())
        arrs["slope"][i] = (t @ rows) / norm if norm > 0 else np.zeros(d)

        fq = rows[:q].mean(axis=0)
        lq = rows[-q:].mean(axis=0)
        arrs["diff_q"][i] = lq - fq

    return run_order, arrs


# ─── helpers ──────────────────────────────────────────────────────────────────

def sep_index(tr: np.ndarray, te: np.ndarray) -> float:
    return float((te.mean() - tr.mean()) / (tr.std() + 1e-10))


def compare(pred_row: np.ndarray) -> str:
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


# ─── pipeline ─────────────────────────────────────────────────────────────────

def fit_models(tr_feats: np.ndarray):
    """Fit GMM + Mahal + LOF on run-level features."""
    gmm = GaussianMixture(
        n_components=2, covariance_type="tied",
        reg_covar=1e-4, n_init=5, max_iter=300, random_state=RANDOM_SEED,
    )
    gmm.fit(tr_feats)

    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_feats)

    lof = LocalOutlierFactor(novelty=True, n_neighbors=10)
    lof.fit(tr_feats)

    return gmm, mah, lof


def score_models(gmm, mah, lof, tr_feats, te_feats):
    gmm_tr = -gmm.score_samples(tr_feats)
    gmm_te = -gmm.score_samples(te_feats)
    mah_tr = -mah.decision_function(tr_feats)
    mah_te = -mah.decision_function(te_feats)
    lof_tr = -lof.decision_function(tr_feats)
    lof_te = -lof.decision_function(te_feats)
    return gmm_tr, gmm_te, mah_tr, mah_te, lof_tr, lof_te


def ensemble_grid(gmm_tr, gmm_te, mah_tr, mah_te, lof_tr, lof_te,
                  te_run_order, test_run_ids):
    """Try all weight combos, return best (by sep)."""
    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg

    zg_te, zg_tr = z(gmm_tr, gmm_te)
    zm_te, zm_tr = z(mah_tr, mah_te)
    zl_te, zl_tr = z(lof_tr, lof_te)

    best = {"sep": -np.inf}
    for wg, wm, wl in WEIGHT_GRID:
        ens_te = wg * zg_te + wm * zm_te + wl * zl_te
        ens_tr = wg * zg_tr + wm * zm_tr + wl * zl_tr
        sep    = sep_index(ens_tr, ens_te)

        thr      = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
        pred_run = (ens_te >= thr).astype(int)
        pred_row = test_run_ids.map(
            pd.Series(pred_run, index=te_run_order)
        ).values
        cmp = compare(pred_row)
        pos = pred_run.mean()

        print(f"    w=({wg:.2f}/{wm:.2f}/{wl:.2f})  sep={sep:9.1f}  pos={pos:.4f}  {cmp}")

        if sep > best["sep"]:
            best.update({"sep": sep, "pred_row": pred_row,
                         "weights": (wg, wm, wl), "cmp": cmp})

    return best


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("데이터 로딩...")
    train_data    = load_train(DATA_PATH)
    test_data     = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler   = fit_scaler(select_features(train_data), scaler_type="standard")
    tr_X_df  = scale_features(select_features(train_data), scaler)
    te_X_df  = scale_features(select_features(test_data),  scaler)
    tr_X     = tr_X_df.values.astype(np.float64)
    te_X     = te_X_df.values.astype(np.float64)

    print("Run 통계 계산 중 (train)...")
    tr_run_order, tr_stats = compute_run_stats(tr_X, train_run_ids.values)
    print("Run 통계 계산 중 (test)...")
    te_run_order, te_stats = compute_run_stats(te_X, test_run_ids.values)

    summary = []

    for fs_name, stat_keys in FEATURE_SETS.items():
        n_dim = len(stat_keys) * 52
        print(f"\n{'='*65}")
        print(f"[{fs_name}]  {n_dim}차원")
        print(f"{'='*65}")

        tr_feats = np.hstack([tr_stats[k] for k in stat_keys])
        te_feats = np.hstack([te_stats[k] for k in stat_keys])

        # Run-level standardize (중요: 피처 유형마다 스케일 다름)
        rl_scaler = StandardScaler()
        tr_feats  = rl_scaler.fit_transform(tr_feats)
        te_feats  = rl_scaler.transform(te_feats)

        gmm, mah, lof = fit_models(tr_feats)
        gmm_tr, gmm_te, mah_tr, mah_te, lof_tr, lof_te = score_models(
            gmm, mah, lof, tr_feats, te_feats
        )
        print(f"  sep →  GMM={sep_index(gmm_tr,gmm_te):.1f}  "
              f"Mahal={sep_index(mah_tr,mah_te):.1f}  "
              f"LOF={sep_index(lof_tr,lof_te):.1f}")

        print("  앙상블 탐색:")
        result = ensemble_grid(gmm_tr, gmm_te, mah_tr, mah_te, lof_tr, lof_te,
                               te_run_order, test_run_ids)

        tag      = fs_name.replace("+", "_")
        out_path = OUTPUT_DIR / f"output_fexp_{tag}.csv"
        save_submission(result["pred_row"], te_X_df.index, out_path)

        summary.append({
            "name": fs_name, "dims": n_dim,
            "sep":  result["sep"],
            "w":    result["weights"],
            "cmp":  result["cmp"],
        })

    # ── 전체 요약 ──
    print("\n" + "=" * 70)
    print("전체 요약 (sep 내림차순)")
    print("=" * 70)
    summary.sort(key=lambda r: r["sep"], reverse=True)
    for r in summary:
        print(f"{r['name']:22s}  {r['dims']:4d}dim  "
              f"sep={r['sep']:9.1f}  w={str(r['w']):20s}  {r['cmp']}")


if __name__ == "__main__":
    main()
