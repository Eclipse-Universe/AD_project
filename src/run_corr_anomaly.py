"""
run_corr_anomaly.py — 센서 간 상관 구조 변화 기반 이상 탐지

핵심 아이디어:
  현재 모델들은 모두 run 평균 벡터 기반 → 52개 센서의 '평균값'만 본다.
  하지만 TEP fault는 센서 간 상관관계를 깬다:
    - 정상 운전: xmv_7↔xmeas_12 (r≈1.0), 수십 쌍의 강한 상관 유지
    - 이상 운전: 특정 센서 쌍의 상관이 붕괴됨

  각 run의 52×52 상관 행렬을 계산해, 정상 train 평균 상관 행렬과의
  Frobenius 거리를 이상 점수로 사용.
  → 평균값은 정상이지만 상관이 깨진 이상(GMM이 놓치는 7개 FN)을 포착 가능.

추가: 각 run의 PCA residual (train PCA subspace 밖 성분의 크기) 도 추가 탐색.

실행: cd /root/AD_project/src && python run_corr_anomaly.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features
from sklearn.mixture import GaussianMixture

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
}

WEIGHT_GRID = [
    (0.45, 0.45, 0.10),
    (0.50, 0.30, 0.20),
    (0.50, 0.40, 0.10),
    (0.40, 0.40, 0.20),
    (0.60, 0.30, 0.10),
    (0.40, 0.50, 0.10),
]


def sep_index(tr, te):
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


# ─── 상관 행렬 기반 이상 점수 ─────────────────────────────────────────────────

def compute_corr_score(X_arr, run_ids_arr, train_mean_corr=None):
    """
    각 run의 52×52 상관 행렬과 train 평균 상관 행렬의 Frobenius 거리를 반환.
    train_mean_corr=None이면 이 데이터로 평균 계산 (train mode).
    """
    groups = {}
    for i, rid in enumerate(run_ids_arr):
        groups.setdefault(int(rid), []).append(i)
    run_order = sorted(groups.keys())

    corr_mats = []
    for rid in run_order:
        rows = X_arr[np.array(groups[rid])]
        if rows.shape[0] < 3:
            corr_mats.append(np.eye(rows.shape[1]))
            continue
        # np.corrcoef on (T, d).T → (d, d)
        c = np.corrcoef(rows.T)
        # NaN (zero-variance column) → 0 off-diagonal, 1 diagonal
        c = np.where(np.isnan(c), np.eye(rows.shape[1]), c)
        corr_mats.append(c)

    corr_mats = np.stack(corr_mats)  # (n_runs, d, d)

    if train_mean_corr is None:
        return run_order, corr_mats, corr_mats.mean(axis=0)

    # Frobenius distance to train mean
    diff   = corr_mats - train_mean_corr[None, :, :]
    scores = np.sqrt((diff ** 2).sum(axis=(1, 2)))
    return run_order, scores


# ─── PCA residual 기반 이상 점수 ──────────────────────────────────────────────

def compute_pca_residual(tr_X, te_X, train_run_ids, test_run_ids, n_components=20):
    """
    각 run의 960 timestep을 train PCA 부분공간에 투영 후 잔차(SPE) run 평균.
    run 단위 mean vector가 아닌 timestep-level residual → 운동 패턴 차이 포착.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(tr_X)

    # train: run별 SPE 평균
    tr_recon = pca.inverse_transform(pca.transform(tr_X))
    tr_spe   = ((tr_X - tr_recon) ** 2).sum(axis=1)  # (250K,)
    tr_sc    = pd.Series(tr_spe).groupby(train_run_ids.values).mean().values

    # test: run별 SPE 평균
    te_recon = pca.inverse_transform(pca.transform(te_X))
    te_spe   = ((te_X - te_recon) ** 2).sum(axis=1)  # (710K,)
    te_sc    = pd.Series(te_spe).groupby(test_run_ids.values).mean()
    te_run_order = te_sc.index.tolist()
    te_sc = te_sc.values

    return tr_sc, te_sc, te_run_order


# ─── 표준 GMM+Mahal run-level 점수 ────────────────────────────────────────────

def compute_gmm_mahal_scores(tr_run_means, te_run_means):
    gmm = GaussianMixture(
        n_components=5, covariance_type="tied",
        reg_covar=1e-6, n_init=10, max_iter=300, random_state=RANDOM_SEED,
    )
    gmm.fit(tr_run_means)
    gmm_tr = -gmm.score_samples(tr_run_means)
    gmm_te = -gmm.score_samples(te_run_means)

    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_run_means)
    mah_tr = -mah.decision_function(tr_run_means)
    mah_te = -mah.decision_function(te_run_means)

    return gmm_tr, gmm_te, mah_tr, mah_te


# ─── 앙상블 ───────────────────────────────────────────────────────────────────

def run_ensemble_2d(sc1_tr, sc1_te, sc2_tr, sc2_te, w1, w2,
                    te_run_order, test_run_ids):
    """2-model Z-score ensemble."""
    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg
    z1_te, z1_tr = z(sc1_tr, sc1_te)
    z2_te, z2_tr = z(sc2_tr, sc2_te)
    ens_te = w1 * z1_te + w2 * z2_te
    ens_tr = w1 * z1_tr + w2 * z2_tr
    sep    = sep_index(ens_tr, ens_te)
    thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
    pred_r = (ens_te >= thr).astype(int)
    pred_row = test_run_ids.map(pd.Series(pred_r, index=te_run_order)).values
    return pred_row, sep


def run_ensemble_3d(s1_tr, s1_te, s2_tr, s2_te, s3_tr, s3_te,
                    w1, w2, w3, te_run_order, test_run_ids):
    """3-model Z-score ensemble."""
    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg
    z1_te, z1_tr = z(s1_tr, s1_te)
    z2_te, z2_tr = z(s2_tr, s2_te)
    z3_te, z3_tr = z(s3_tr, s3_te)
    ens_te = w1 * z1_te + w2 * z2_te + w3 * z3_te
    ens_tr = w1 * z1_tr + w2 * z2_tr + w3 * z3_tr
    sep    = sep_index(ens_tr, ens_te)
    thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
    pred_r = (ens_te >= thr).astype(int)
    pred_row = test_run_ids.map(pd.Series(pred_r, index=te_run_order)).values
    return pred_row, sep


# ─── main ─────────────────────────────────────────────────────────────────────

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

    # ── run 평균 벡터 (기존 방식) ──
    print("Run 평균 벡터 계산 중...")
    tr_run_means = tr_X_df.groupby(train_run_ids.values).mean().values  # (500, 52)
    te_run_vecs  = te_X_df.groupby(test_run_ids.values).mean()
    te_run_order_mean = te_run_vecs.index.tolist()
    te_run_means = te_run_vecs.values  # (740, 52)

    # ── GMM + Mahal (기존 최고 모델, 참조용) ──
    print("GMM+Mahal 학습 중 (참조)...")
    gmm_tr, gmm_te, mah_tr, mah_te = compute_gmm_mahal_scores(
        tr_run_means, te_run_means
    )
    print(f"  GMM sep={sep_index(gmm_tr,gmm_te):.1f}  "
          f"Mahal sep={sep_index(mah_tr,mah_te):.1f}")

    # ── A. 상관 구조 변화 점수 ──────────────────────────────────────────────
    print("\n[A] 상관 행렬 Frobenius 거리 계산 중...")
    _, tr_corr_mats, train_mean_corr = compute_corr_score(tr_X, train_run_ids.values)
    te_corr_run_order, corr_scores = compute_corr_score(
        te_X, test_run_ids.values, train_mean_corr
    )
    # train에도 같은 방법 적용 (Z-norm 기준)
    _, tr_corr_scores_raw = compute_corr_score(tr_X, train_run_ids.values, train_mean_corr)
    corr_tr = tr_corr_scores_raw  # (500,)
    corr_te = corr_scores          # (740,)

    print(f"  Corr-Frob sep={sep_index(corr_tr, corr_te):.2f}")
    print(f"  train corr score: mean={corr_tr.mean():.3f} std={corr_tr.std():.3f}")
    print(f"  test  corr score: mean={corr_te.mean():.3f} std={corr_te.std():.3f}")

    # Corr 단독 예측
    thr      = np.quantile(corr_te, 1 - RUN_CONTAMINATION)
    pred_run = (corr_te >= thr).astype(int)
    pred_row_corr = test_run_ids.map(
        pd.Series(pred_run, index=te_corr_run_order)
    ).values
    print(f"  Corr 단독: {compare(pred_row_corr)}")

    # ── B. PCA residual 점수 ─────────────────────────────────────────────────
    print("\n[B] PCA Residual(SPE) 계산 중...")
    for n_pc in [10, 20, 30]:
        spe_tr, spe_te, spe_run_order = compute_pca_residual(
            tr_X, te_X, train_run_ids, test_run_ids, n_components=n_pc
        )
        sep_spe = sep_index(spe_tr, spe_te)
        thr      = np.quantile(spe_te, 1 - RUN_CONTAMINATION)
        pred_run = (spe_te >= thr).astype(int)
        pred_row_spe = test_run_ids.map(
            pd.Series(pred_run, index=spe_run_order)
        ).values
        cmp = compare(pred_row_spe)
        print(f"  PCA(k={n_pc:2d}) sep={sep_spe:.1f}  {cmp}")
    # k=30 SPE 사용 (가장 높은 sep)
    spe_tr, spe_te, spe_run_order = compute_pca_residual(
        tr_X, te_X, train_run_ids, test_run_ids, n_components=30
    )

    # ── C. 앙상블 탐색: GMM + Mahal + Corr ──────────────────────────────────
    print("\n[C] GMM+Mahal+Corr 앙상블:")
    # corr_te/tr는 te_corr_run_order 기준 → 평균 벡터 run_order와 동일 (둘 다 sorted)
    assert te_corr_run_order == te_run_order_mean, \
        "run order mismatch — check groupby sort"

    best_c = {"sep": -np.inf}
    for wg, wm, wc in WEIGHT_GRID:
        pred_row, sep = run_ensemble_3d(
            gmm_tr, gmm_te, mah_tr, mah_te, corr_tr, corr_te,
            wg, wm, wc, te_run_order_mean, test_run_ids
        )
        cmp = compare(pred_row)
        print(f"  w=({wg:.2f}/{wm:.2f}/{wc:.2f})  sep={sep:9.1f}  {cmp}")
        if sep > best_c["sep"]:
            best_c.update({"sep": sep, "pred_row": pred_row, "w": (wg,wm,wc), "cmp": cmp})

    save_submission(best_c["pred_row"], te_X_df.index,
                    OUTPUT_DIR / "output_fexp_corr.csv")
    print(f"  → 최적: w={best_c['w']} sep={best_c['sep']:.1f}  {best_c['cmp']}")

    # ── D. 앙상블 탐색: GMM + Mahal + SPE ───────────────────────────────────
    print("\n[D] GMM+Mahal+SPE(k=30) 앙상블:")
    best_s = {"sep": -np.inf}
    for wg, wm, ws in WEIGHT_GRID:
        pred_row, sep = run_ensemble_3d(
            gmm_tr, gmm_te, mah_tr, mah_te, spe_tr, spe_te,
            wg, wm, ws, spe_run_order, test_run_ids
        )
        cmp = compare(pred_row)
        print(f"  w=({wg:.2f}/{wm:.2f}/{ws:.2f})  sep={sep:9.1f}  {cmp}")
        if sep > best_s["sep"]:
            best_s.update({"sep": sep, "pred_row": pred_row, "w": (wg,wm,ws), "cmp": cmp})

    save_submission(best_s["pred_row"], te_X_df.index,
                    OUTPUT_DIR / "output_fexp_spe.csv")
    print(f"  → 최적: w={best_s['w']} sep={best_s['sep']:.1f}  {best_s['cmp']}")

    # ── E. 4모델: GMM+Mahal+Corr+SPE ────────────────────────────────────────
    print("\n[E] GMM+Mahal+Corr+SPE 4모델 앙상블:")
    W4_GRID = [
        (0.40, 0.35, 0.15, 0.10),
        (0.40, 0.35, 0.10, 0.15),
        (0.45, 0.40, 0.10, 0.05),
        (0.35, 0.35, 0.20, 0.10),
        (0.35, 0.35, 0.10, 0.20),
        (0.40, 0.30, 0.20, 0.10),
    ]
    def z(tr, te):
        mu, sg = tr.mean(), tr.std() + 1e-10
        return (te - mu) / sg, (tr - mu) / sg

    zg_te, zg_tr = z(gmm_tr, gmm_te)
    zm_te, zm_tr = z(mah_tr, mah_te)
    zc_te, zc_tr = z(corr_tr, corr_te)
    zs_te, zs_tr = z(spe_tr,  spe_te)

    best_4 = {"sep": -np.inf}
    for wg, wm, wc, ws in W4_GRID:
        ens_te = wg*zg_te + wm*zm_te + wc*zc_te + ws*zs_te
        ens_tr = wg*zg_tr + wm*zm_tr + wc*zc_tr + ws*zs_tr
        sep    = sep_index(ens_tr, ens_te)
        thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
        pred_r = (ens_te >= thr).astype(int)
        pred_row = test_run_ids.map(
            pd.Series(pred_r, index=te_run_order_mean)
        ).values
        cmp = compare(pred_row)
        print(f"  w=({wg:.2f}/{wm:.2f}/{wc:.2f}/{ws:.2f})  sep={sep:9.1f}  {cmp}")
        if sep > best_4["sep"]:
            best_4.update({"sep": sep, "pred_row": pred_row,
                           "w": (wg,wm,wc,ws), "cmp": cmp})

    save_submission(best_4["pred_row"], te_X_df.index,
                    OUTPUT_DIR / "output_fexp_4model.csv")

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("요약")
    print("="*70)
    print(f"  Corr-Frob 단독:          {compare(pred_row_corr)}")
    print(f"  GMM+Mahal+Corr:   sep={best_c['sep']:9.1f}  {best_c['cmp']}")
    print(f"  GMM+Mahal+SPE:    sep={best_s['sep']:9.1f}  {best_s['cmp']}")
    print(f"  GMM+Mahal+C+S:    sep={best_4['sep']:9.1f}  {best_4['cmp']}")
    print("\n저장 파일: output_fexp_corr.csv / output_fexp_spe.csv / output_fexp_4model.csv")


if __name__ == "__main__":
    main()
