"""
run_spe_finetune.py — SPE k 세밀 탐색 및 앙상블 가중치 최적화 (Exp32 후보)

배경:
  Exp30 (GMM+Mahal+SPE k=30, 가중치 0.6/0.3/0.1) → F1=0.9707 신규 최고.
  탐색한 k: {10, 20, 30} → k=30이 sep 최대.
  이번 실험: k={15,20,25,30,35,40,50}을 더 세밀하게 탐색 + 가중치 그리드 확장.
  추가: LOF를 4번째 앙상블 성분으로 추가하는 효과도 확인.

실행: cd /root/AD_project/src && python run_spe_finetune.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42
TEST_TS           = 960

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
    "Exp30": str(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv"),
}

SPE_K_GRID = [15, 20, 25, 30, 35, 40, 50]

# 3-모델 가중치 (GMM, Mahal, SPE)
WEIGHT_GRID_3 = [
    (0.60, 0.30, 0.10),
    (0.50, 0.30, 0.20),
    (0.50, 0.35, 0.15),
    (0.50, 0.40, 0.10),
    (0.45, 0.45, 0.10),
    (0.45, 0.35, 0.20),
    (0.40, 0.40, 0.20),
    (0.55, 0.30, 0.15),
    (0.55, 0.35, 0.10),
    (0.40, 0.30, 0.30),
]

# 4-모델 가중치 (GMM, Mahal, SPE, LOF)
WEIGHT_GRID_4 = [
    (0.50, 0.25, 0.15, 0.10),
    (0.45, 0.30, 0.15, 0.10),
    (0.45, 0.25, 0.20, 0.10),
    (0.40, 0.30, 0.20, 0.10),
    (0.50, 0.25, 0.10, 0.15),
    (0.45, 0.30, 0.10, 0.15),
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


def compute_run_spe(tr_X: np.ndarray, te_X: np.ndarray,
                    train_run_ids, test_run_ids, k: int):
    """행 단위 PCA → 행별 SPE → run 평균 (train/test)."""
    pca = PCA(n_components=k, random_state=RANDOM_SEED)
    pca.fit(tr_X)

    def row_spe(X):
        X_hat = pca.inverse_transform(pca.transform(X))
        return ((X - X_hat) ** 2).sum(axis=1)

    tr_spe_row = row_spe(tr_X)
    te_spe_row = row_spe(te_X)

    tr_sc = pd.Series(tr_spe_row, index=train_run_ids.values).groupby(level=0).mean()
    te_sc = pd.Series(te_spe_row, index=test_run_ids.values).groupby(level=0).mean()
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

    # ── 기준 모델: GMM + Mahal (run 평균 벡터) ─────────────────────────────
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
    print(f"  GMM sep={sep_index(gmm_tr, gmm_te):.0f}")

    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_rvec)
    mah_tr = -mah.decision_function(tr_rvec)
    mah_te = -mah.decision_function(te_rvec)
    print(f"  Mahal sep={sep_index(mah_tr, mah_te):.0f}")

    zg_te, zg_tr = z(gmm_tr, gmm_te)
    zm_te, zm_tr = z(mah_tr, mah_te)

    # ── LOF (4-모델 앙상블용) ───────────────────────────────────────────────
    print("LOF 스코어 계산 중...")
    lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
    lof.fit(tr_rvec)
    lof_tr = -lof.decision_function(tr_rvec)
    lof_te = -lof.decision_function(te_rvec)
    print(f"  LOF sep={sep_index(lof_tr, lof_te):.1f}")
    zl_te, zl_tr = z(lof_tr, lof_te)

    # ── SPE k 세밀 탐색 ────────────────────────────────────────────────────
    best_overall = {"sep": -np.inf, "row": None, "tag": "", "cmp": ""}

    print("\n" + "=" * 70)
    print("SPE k 세밀 탐색")
    print("=" * 70)

    for k in SPE_K_GRID:
        tr_spe, te_spe = compute_run_spe(tr_X, te_X, train_run_ids, test_run_ids, k)

        # run 순서 맞추기
        spe_tr_arr = tr_spe.reindex(tr_run_order).values
        spe_te_arr = te_spe.reindex(te_run_order).values
        spe_sep = sep_index(spe_tr_arr, spe_te_arr)

        zs_te, zs_tr = z(spe_tr_arr, spe_te_arr)

        # 3-모델 앙상블 탐색
        best_3_sep, best_3_row, best_3_w, best_3_cmp = -np.inf, None, None, None
        for wg, wm, ws in WEIGHT_GRID_3:
            ens_te = wg * zg_te + wm * zm_te + ws * zs_te
            ens_tr = wg * zg_tr + wm * zm_tr + ws * zs_tr
            sep_e  = sep_index(ens_tr, ens_te)
            thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
            pred   = (ens_te >= thr).astype(int)
            row    = test_run_ids.map(pd.Series(pred, index=te_run_order)).values
            if sep_e > best_3_sep:
                best_3_sep, best_3_row, best_3_w, best_3_cmp = sep_e, row, (wg, wm, ws), compare(row)

        # 4-모델 앙상블 탐색 (GMM+Mahal+SPE+LOF)
        best_4_sep, best_4_row, best_4_w, best_4_cmp = -np.inf, None, None, None
        for wg, wm, ws, wl in WEIGHT_GRID_4:
            ens_te = wg * zg_te + wm * zm_te + ws * zs_te + wl * zl_te
            ens_tr = wg * zg_tr + wm * zm_tr + ws * zs_tr + wl * zl_tr
            sep_e  = sep_index(ens_tr, ens_te)
            thr    = np.quantile(ens_te, 1 - RUN_CONTAMINATION)
            pred   = (ens_te >= thr).astype(int)
            row    = test_run_ids.map(pd.Series(pred, index=te_run_order)).values
            if sep_e > best_4_sep:
                best_4_sep, best_4_row, best_4_w, best_4_cmp = sep_e, row, (wg, wm, ws, wl), compare(row)

        # 최고 성능 선택 (sep 기준)
        if best_3_sep >= best_4_sep:
            tag = f"GMM-Mahal-SPE{k}"
            row = best_3_row
            cmp = best_3_cmp
            sep = best_3_sep
            wstr = str(best_3_w)
        else:
            tag = f"GMM-Mahal-SPE{k}-LOF"
            row = best_4_row
            cmp = best_4_cmp
            sep = best_4_sep
            wstr = str(best_4_w)

        save_submission(row, te_X_df.index, OUTPUT_DIR / f"output_spe_{k}.csv")

        print(f"  k={k:2d}  SPE_sep={spe_sep:7.0f}  "
              f"3-model: {best_3_cmp} (sep={best_3_sep:.1f}, w={best_3_w})  |  "
              f"4-model: {best_4_cmp} (sep={best_4_sep:.1f})")

        if sep > best_overall["sep"]:
            best_overall = {"sep": sep, "row": row, "tag": tag, "cmp": cmp, "k": k, "w": wstr}

    # ── 최종 최고 파일 저장 ────────────────────────────────────────────────
    best_tag = best_overall["tag"]
    best_k   = best_overall["k"]
    best_row = best_overall["row"]
    save_submission(best_row, te_X_df.index,
                    OUTPUT_DIR / f"output_exp32({best_tag}).csv")

    print("\n" + "=" * 70)
    print(f"최종 최고: k={best_k}, 모델={best_tag}")
    print(f"  sep={best_overall['sep']:.1f}, w={best_overall['w']}")
    print(f"  {best_overall['cmp']}")
    print(f"  → output_exp32({best_tag}).csv 저장")


if __name__ == "__main__":
    main()
