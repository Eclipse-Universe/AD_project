"""GMM + KMeans-Mahal + IsolationForest 앙상블.

Exp25(GMM+Mahal+LOF, F1=0.9623) 이후 IF로 LOF를 대체/추가.
LOF는 row-level 집계 모델 → run 판별 기여가 약함(+0.0003).
IF는 Exp12(run-level, F1=0.8870)에서 이미 run 벡터 직접 모델링이 가능함을 확인.
경로 길이 기반 점수는 GMM(밀도)/Mahal(거리)과 원리적으로 다름 → 상보성 기대.

두 가지 구성 비교:
  A. GMM + Mahal + IF   (LOF 제거, IF 대체)
  B. GMM + Mahal + IF + LOF  (4모델)

실행: cd src && python run_ensemble_if.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector, train_lof
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

# 3모델 (GMM, Mahal, IF) 가중치 그리드
WEIGHT_GRID_3 = [
    (0.45, 0.45, 0.10),
    (0.40, 0.40, 0.20),
    (0.35, 0.35, 0.30),
    (0.50, 0.30, 0.20),
    (0.50, 0.25, 0.25),
    (0.33, 0.33, 0.34),
]

# 4모델 (GMM, Mahal, IF, LOF) 가중치 — IF, LOF를 각 10~20%로
WEIGHT_GRID_4 = [
    (0.40, 0.40, 0.10, 0.10),
    (0.40, 0.35, 0.15, 0.10),
    (0.38, 0.38, 0.12, 0.12),
    (0.35, 0.35, 0.20, 0.10),
]


def build_run_vectors(X, run_ids):
    return X.groupby(run_ids.values).mean()


def z_norm(train_s, test_s):
    mu, sigma = train_s.mean(), train_s.std()
    return (train_s - mu) / sigma, (test_s - mu) / sigma


def predict(ens_test, run_index, test_run_ids):
    thr = np.quantile(ens_test, 1 - RUN_CONTAMINATION)
    pred_run = pd.Series((ens_test >= thr).astype(int), index=run_index)
    return test_run_ids.map(pred_run).values


def compare(pred_rows, ref_files):
    pred_run = np.array(pred_rows).reshape(-1, 960)[:, 0]
    parts = []
    for name, path in ref_files.items():
        if not Path(path).exists():
            continue
        ref_run  = pd.read_csv(path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree    = (pred_run == ref_run).sum()
        me_only  = ((pred_run == 1) & (ref_run == 0)).sum()
        ref_only = ((pred_run == 0) & (ref_run == 1)).sum()
        parts.append(f"{name}: agree={agree}/740 | mine={me_only} | {name}={ref_only}")
    return "  " + "  ".join(parts)


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)
    train_vecs = build_run_vectors(train_X, train_run_ids)
    test_vecs  = build_run_vectors(test_X,  test_run_ids)

    ref_files = {
        "Ens25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
        "Ens24": str(OUTPUT_DIR / "output_exp24(Ensemble-GMM-Mahal).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
    }

    # ── GMM (Exp21/24/25와 동일)
    print("[GMM 학습 중]")
    gmm = GaussianMixture(n_components=5, covariance_type="tied",
                          reg_covar=1e-6, random_state=RANDOM_SEED,
                          max_iter=300, n_init=5)
    gmm.fit(train_vecs.values)
    gmm_tr = -gmm.score_samples(train_vecs.values)
    gmm_te = -gmm.score_samples(test_vecs.values)

    # ── Mahal (Exp17/24/25와 동일)
    print("[Mahal 학습 중]")
    mahal = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mahal.fit(train_X)
    mahal_tr = pd.Series(-mahal.decision_function(train_X), index=train_X.index)\
                 .groupby(train_run_ids.values).mean().values
    mahal_te = pd.Series(-mahal.decision_function(test_X), index=test_X.index)\
                 .groupby(test_run_ids.values).mean().values

    # ── IsolationForest — run-level (Exp12와 동일 구조, 단 run 벡터 직접 학습)
    print("[IsolationForest 학습 중]")
    iforest = IsolationForest(n_estimators=200, max_samples="auto",
                              random_state=RANDOM_SEED, n_jobs=-1)
    iforest.fit(train_vecs.values)
    # decision_function: 높을수록 정상 → 부호 반전해서 높을수록 이상
    if_tr = -iforest.decision_function(train_vecs.values)
    if_te = -iforest.decision_function(test_vecs.values)

    # ── LOF (Exp25와 동일, 4모델 앙상블용)
    print("[LOF 학습 중]")
    lof = train_lof(train_X, n_neighbors=20)
    lof_tr = pd.Series(-lof.decision_function(train_X.values), index=train_X.index)\
               .groupby(train_run_ids.values).mean().values
    lof_te = pd.Series(-lof.decision_function(test_X.values), index=test_X.index)\
               .groupby(test_run_ids.values).mean().values

    # ── Z-score 정규화
    gmm_tr_n,   gmm_te_n   = z_norm(gmm_tr,   gmm_te)
    mahal_tr_n, mahal_te_n = z_norm(mahal_tr, mahal_te)
    if_tr_n,    if_te_n    = z_norm(if_tr,    if_te)
    lof_tr_n,   lof_te_n   = z_norm(lof_tr,   lof_te)

    print(f"\n  GMM   μ={gmm_tr.mean():.3f} σ={gmm_tr.std():.3f}")
    print(f"  Mahal μ={mahal_tr.mean():.3f} σ={mahal_tr.std():.3f}")
    print(f"  IF    μ={if_tr.mean():.3f} σ={if_tr.std():.3f}")
    print(f"  LOF   μ={lof_tr.mean():.3f} σ={lof_tr.std():.3f}")

    run_index = test_vecs.index

    # ── 3모델: GMM + Mahal + IF
    print("\n" + "=" * 65)
    print("3모델 앙상블: GMM + Mahal + IF")
    print("=" * 65)

    best3_sep, best3_w, best3_pred = -np.inf, None, None
    for w_gmm, w_mahal, w_if in WEIGHT_GRID_3:
        ens_tr = w_gmm*gmm_tr_n + w_mahal*mahal_tr_n + w_if*if_tr_n
        ens_te = w_gmm*gmm_te_n + w_mahal*mahal_te_n + w_if*if_te_n
        sep    = (ens_te.mean() - ens_tr.mean()) / ens_tr.std()
        pred   = predict(ens_te, run_index, test_run_ids)

        tag = f"g{w_gmm}_m{w_mahal}_if{w_if}"
        save_submission(pred, test_X.index, OUTPUT_DIR / f"output_ens_if3_{tag}.csv")

        print(f"\n[GMM={w_gmm} Mahal={w_mahal} IF={w_if}]  sep={sep:.3f}  pos={pred.mean():.4f}")
        print(compare(pred, ref_files))

        if sep > best3_sep:
            best3_sep, best3_w, best3_pred = sep, (w_gmm, w_mahal, w_if), pred

    print(f"\n→ 3모델(IF) 최적: GMM={best3_w[0]} Mahal={best3_w[1]} IF={best3_w[2]}  sep={best3_sep:.3f}")

    # ── 4모델: GMM + Mahal + IF + LOF
    print("\n" + "=" * 65)
    print("4모델 앙상블: GMM + Mahal + IF + LOF")
    print("=" * 65)

    best4_sep, best4_w, best4_pred = -np.inf, None, None
    for w_gmm, w_mahal, w_if, w_lof in WEIGHT_GRID_4:
        ens_tr = w_gmm*gmm_tr_n + w_mahal*mahal_tr_n + w_if*if_tr_n + w_lof*lof_tr_n
        ens_te = w_gmm*gmm_te_n + w_mahal*mahal_te_n + w_if*if_te_n + w_lof*lof_te_n
        sep    = (ens_te.mean() - ens_tr.mean()) / ens_tr.std()
        pred   = predict(ens_te, run_index, test_run_ids)

        tag = f"g{w_gmm}_m{w_mahal}_if{w_if}_l{w_lof}"
        save_submission(pred, test_X.index, OUTPUT_DIR / f"output_ens_if4_{tag}.csv")

        print(f"\n[GMM={w_gmm} Mahal={w_mahal} IF={w_if} LOF={w_lof}]  sep={sep:.3f}  pos={pred.mean():.4f}")
        print(compare(pred, ref_files))

        if sep > best4_sep:
            best4_sep, best4_w, best4_pred = sep, (w_gmm, w_mahal, w_if, w_lof), pred

    print(f"\n→ 4모델 최적: GMM={best4_w[0]} Mahal={best4_w[1]} IF={best4_w[2]} LOF={best4_w[3]}  sep={best4_sep:.3f}")


if __name__ == "__main__":
    main()
