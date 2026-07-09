"""
run_autocorr.py — 자기상관(ACF) + 고차 통계 기반 동적 피처 이상 탐지 (Exp33 후보)

배경:
  GMM+Mahal+SPE(k=30)으로 탐지하지 못하는 7개 구조적 FN run:
    run107, run176, run274, run280, run287, run549, run685
  이 run들은 mean, std, slope, diff_q, p10/p90, Corr-Frob, SPE 모두 탐지 실패.
  즉, 단면(cross-sectional) 통계로는 정상처럼 보임.

  가설: 이상이 '시간 구조(temporal dynamics)'에서만 나타날 수 있음.
    - 정상 운전: 각 센서 신호가 안정적 자기상관 패턴 (정상 AR 구조)
    - 이상 운전: 진동 주파수 변화 / 자기상관 패턴 붕괴

  탐색:
    A. ACF(자기상관) 피처: lag {1,2,5,10,20,50}에서 52개 센서의 자기상관 값
       → 52×6 = 312 피처. run 내 시계열 temporal 구조 요약.
    B. 고차 통계: 왜도(skewness), 첨도(kurtosis) per sensor per run
       → 52×2 = 104 피처. 분포 형태 변화 탐지.
    C. A+B 결합: 52×8 = 416 피처.
    D. 각 피처 세트를 GMM+Mahal+SPE 앙상블에 Z-score로 추가.

실행: cd /root/AD_project/src && python run_autocorr.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42
TEST_TS           = 960
TRAIN_TS          = 500

ACF_LAGS = [1, 2, 5, 10, 20, 50]

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
    "Exp30": str(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv"),
}

WEIGHT_GRID = [
    (0.55, 0.25, 0.10, 0.10),   # GMM, Mahal, SPE, dynfeat
    (0.50, 0.25, 0.10, 0.15),
    (0.50, 0.25, 0.15, 0.10),
    (0.45, 0.25, 0.15, 0.15),
    (0.40, 0.25, 0.15, 0.20),
    (0.50, 0.20, 0.10, 0.20),
    (0.45, 0.20, 0.10, 0.25),
    (0.55, 0.25, 0.08, 0.12),
    (0.60, 0.25, 0.08, 0.07),
    # baseline (Exp30 구성): SPE만, dynfeat 없음
    (0.60, 0.30, 0.10, 0.00),
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


def compute_acf_at_lag(x: np.ndarray, lag: int) -> float:
    """1D 시계열의 lag k 자기상관 계수."""
    n = len(x)
    if n <= lag:
        return 0.0
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float((xc[:n - lag] * xc[lag:]).sum() / denom)


def compute_run_dynamic_features(X_arr: np.ndarray, run_ids_arr: np.ndarray,
                                  lags=None, include_higher_order=True):
    """
    각 run의 동적 피처 벡터 계산.
    반환: (run_order list, feature_matrix (n_runs, d_feat))
    """
    if lags is None:
        lags = ACF_LAGS

    groups = {}
    for i, rid in enumerate(run_ids_arr):
        groups.setdefault(int(rid), []).append(i)
    run_order = sorted(groups.keys())
    n_runs = len(run_order)
    n_sensors = X_arr.shape[1]

    # ACF 피처: (n_runs, n_sensors × n_lags)
    n_lags = len(lags)
    acf_mat = np.zeros((n_runs, n_sensors * n_lags))

    # 고차 통계: (n_runs, n_sensors × 2)  [skewness, kurtosis]
    horder_mat = np.zeros((n_runs, n_sensors * 2))

    for i, rid in enumerate(run_order):
        rows = X_arr[np.array(groups[rid])]   # (T, 52)
        T = len(rows)
        for j in range(n_sensors):
            sig = rows[:, j]
            # ACF
            for li, lag in enumerate(lags):
                acf_mat[i, j * n_lags + li] = compute_acf_at_lag(sig, lag)
            # 고차 통계
            if include_higher_order:
                horder_mat[i, j * 2]     = float(scipy_stats.skew(sig))
                horder_mat[i, j * 2 + 1] = float(scipy_stats.kurtosis(sig))

    if include_higher_order:
        feat_mat = np.hstack([acf_mat, horder_mat])
    else:
        feat_mat = acf_mat

    return run_order, feat_mat


def compute_run_spe(tr_X, te_X, train_run_ids, test_run_ids, k=30):
    pca = PCA(n_components=k, random_state=RANDOM_SEED)
    pca.fit(tr_X)
    def spe(X):
        return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)
    tr_sc = pd.Series(spe(tr_X), index=train_run_ids.values).groupby(level=0).mean()
    te_sc = pd.Series(spe(te_X), index=test_run_ids.values).groupby(level=0).mean()
    return tr_sc, te_sc


def z(tr_arr, te_arr):
    mu, sg = tr_arr.mean(), tr_arr.std() + 1e-10
    return (te_arr - mu) / sg, (tr_arr - mu) / sg


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

    # ── 기준 모델: GMM + Mahal (run 평균) ──────────────────────────────────
    print("GMM + Mahal 기준 스코어 계산 중...")
    tr_run_vecs = tr_X_df.groupby(train_run_ids.values).mean()
    te_run_vecs = te_X_df.groupby(test_run_ids.values).mean()
    tr_run_order = tr_run_vecs.index.tolist()
    te_run_order = te_run_vecs.index.tolist()
    tr_rvec = tr_run_vecs.values
    te_rvec = te_run_vecs.values

    gmm = GaussianMixture(
        n_components=2, covariance_type="tied",
        reg_covar=1e-4, n_init=10, max_iter=300, random_state=RANDOM_SEED,
    )
    gmm.fit(tr_rvec)
    gmm_tr = -gmm.score_samples(tr_rvec)
    gmm_te = -gmm.score_samples(te_rvec)

    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_rvec)
    mah_tr = -mah.decision_function(tr_rvec)
    mah_te = -mah.decision_function(te_rvec)

    print(f"  GMM sep={sep_index(gmm_tr, gmm_te):.0f}  Mahal sep={sep_index(mah_tr, mah_te):.0f}")

    zg_te, zg_tr = z(gmm_tr, gmm_te)
    zm_te, zm_tr = z(mah_tr, mah_te)

    # ── SPE k=30 (Exp30 기준) ──────────────────────────────────────────────
    print("PCA-SPE (k=30) 계산 중...")
    tr_spe, te_spe = compute_run_spe(tr_X, te_X, train_run_ids, test_run_ids, k=30)
    spe_tr = tr_spe.reindex(tr_run_order).values
    spe_te = te_spe.reindex(te_run_order).values
    zs_te, zs_tr = z(spe_tr, spe_te)
    print(f"  SPE sep={sep_index(spe_tr, spe_te):.0f}")

    # Exp30 baseline 확인
    ens30_te = 0.6 * zg_te + 0.3 * zm_te + 0.1 * zs_te
    ens30_tr = 0.6 * zg_tr + 0.3 * zm_tr + 0.1 * zs_tr
    thr30    = np.quantile(ens30_te, 1 - RUN_CONTAMINATION)
    pred30   = (ens30_te >= thr30).astype(int)
    row30    = test_run_ids.map(pd.Series(pred30, index=te_run_order)).values
    print(f"  Exp30 재현: {compare(row30)}")

    # ── 동적 피처 세트 ─────────────────────────────────────────────────────
    feat_sets = {
        "ACF":          (ACF_LAGS, False),
        "ACF+HOrder":   (ACF_LAGS, True),
        "ACF_short":    ([1, 2, 5, 10], False),
        "ACF_long":     ([10, 20, 50, 100], False),
    }

    print("\n" + "=" * 70)
    print("동적 피처 탐색")
    print("=" * 70)

    best_overall = {"sep": -np.inf, "row": None, "tag": "", "cmp": ""}

    for feat_name, (lags, higher_order) in feat_sets.items():
        print(f"\n[{feat_name}] 계산 중 (lags={lags}, higher_order={higher_order})...")

        tr_run_ord_d, tr_dfeat_raw = compute_run_dynamic_features(
            tr_X, train_run_ids.values, lags=lags, include_higher_order=higher_order
        )
        te_run_ord_d, te_dfeat_raw = compute_run_dynamic_features(
            te_X, test_run_ids.values, lags=lags, include_higher_order=higher_order
        )

        # GMM(k=2)로 동적 피처 이상 점수 단독 계산
        from sklearn.preprocessing import StandardScaler
        sc_d = StandardScaler()
        tr_dfeat = sc_d.fit_transform(tr_dfeat_raw)
        te_dfeat = sc_d.transform(te_dfeat_raw)

        gmm_d = GaussianMixture(
            n_components=2, covariance_type="tied",
            reg_covar=1e-4, n_init=5, max_iter=300, random_state=RANDOM_SEED,
        )
        gmm_d.fit(tr_dfeat)
        dyn_tr_raw = -gmm_d.score_samples(tr_dfeat)
        dyn_te_raw = -gmm_d.score_samples(te_dfeat)
        dyn_sep = sep_index(dyn_tr_raw, dyn_te_raw)

        # 단독 예측
        thr_d  = np.quantile(dyn_te_raw, 1 - RUN_CONTAMINATION)
        pred_d = (dyn_te_raw >= thr_d).astype(int)
        row_d  = test_run_ids.map(pd.Series(pred_d, index=te_run_ord_d)).values
        cmp_d  = compare(row_d)
        print(f"  단독 sep={dyn_sep:.1f}  {cmp_d}")

        # Z-score 정규화 후 앙상블
        zd_te, zd_tr = z(dyn_tr_raw, dyn_te_raw)

        best_ens_sep, best_ens_row, best_ens_w, best_ens_cmp = -np.inf, None, None, None
        for wg, wm, ws, wd in WEIGHT_GRID:
            ens_te = wg * zg_te + wm * zm_te + ws * zs_te + wd * zd_te
            ens_tr = wg * zg_tr + wm * zm_tr + ws * zs_tr + wd * zd_tr
            sep_e  = sep_index(ens_tr, ens_te)
            thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
            pred   = (ens_te >= thr).astype(int)
            row    = test_run_ids.map(pd.Series(pred, index=te_run_order)).values
            cmp    = compare(row)
            if sep_e > best_ens_sep:
                best_ens_sep, best_ens_row, best_ens_w, best_ens_cmp = sep_e, row, (wg, wm, ws, wd), cmp

        tag = f"GMM-Mahal-SPE30-{feat_name}"
        save_submission(best_ens_row, te_X_df.index,
                        OUTPUT_DIR / f"output_autocorr_{feat_name}.csv")

        print(f"  앙상블 sep={best_ens_sep:.1f}  w={best_ens_w}  {best_ens_cmp}")

        if best_ens_sep > best_overall["sep"]:
            best_overall = {
                "sep": best_ens_sep, "row": best_ens_row,
                "tag": tag, "cmp": best_ens_cmp, "w": best_ens_w,
                "feat": feat_name,
            }

    # ── 최고 결과 저장 ──────────────────────────────────────────────────────
    best_tag  = best_overall["tag"]
    best_feat = best_overall["feat"]
    save_submission(best_overall["row"], te_X_df.index,
                    OUTPUT_DIR / f"output_exp33({best_tag}).csv")

    print("\n" + "=" * 70)
    print(f"최종 최고: feat={best_feat}, 모델={best_tag}")
    print(f"  sep={best_overall['sep']:.1f}, w={best_overall['w']}")
    print(f"  {best_overall['cmp']}")
    print(f"  → output_exp33({best_tag}).csv 저장")


if __name__ == "__main__":
    main()
