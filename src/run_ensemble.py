"""GMM + KMeans-Mahal 앙상블 이상 탐지 — 가중합 alpha 탐색.

배경:
  GMM-tied  (Exp21, F1=0.9372): 24개 run을 Mahal이 못 잡고 GMM만 탐지
  KMeans-Mahal (Exp17, F1=0.9277): 24개 run을 GMM이 못 잡고 Mahal만 탐지
  → 두 모델의 점수를 정규화 후 가중합해 각자의 blind spot 보완

점수 방향 및 정규화:
  모두 "높을수록 이상"으로 통일
  - GMM   : -score_samples()   (부호 반전)
  - Mahal : -decision_function() = min_mahal_dist

  Z-score 정규화 (훈련 run 점수 기준):
  z = (score - μ_train) / σ_train
  → 서로 다른 스케일(로그우도 vs 거리)을 표준 단위로 통일

앙상블:
  ensemble = α × z_gmm + (1-α) × z_mahal
  α 탐색: {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
  상위 RUN_CONTAMINATION = 32%를 이상 run으로 판정

모델 파라미터 (Exp17, Exp21과 동일):
  GMM   : n_components=5, covariance_type='tied', reg_covar=1e-6, n_init=5
  Mahal : n_clusters=50, 행 단위 학습 후 run 평균 집계

실행: cd src && python run_ensemble.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42
ALPHA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def build_run_vectors(X: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    return X.groupby(run_ids.values).mean()


def z_normalize(train_scores: np.ndarray, test_scores: np.ndarray):
    """훈련 run 점수를 기준으로 Z-score 정규화."""
    mu    = train_scores.mean()
    sigma = train_scores.std()
    return (train_scores - mu) / sigma, (test_scores - mu) / sigma


def predict_from_ensemble(ensemble_test: np.ndarray, run_index, test_run_ids, test_X_index):
    """앙상블 점수 상위 32%를 이상 판정, run→행 단위 예측 반환."""
    threshold = np.quantile(ensemble_test, 1 - RUN_CONTAMINATION)
    pred_run  = pd.Series((ensemble_test >= threshold).astype(int), index=run_index)
    return test_run_ids.map(pred_run).values


def compare_refs(pred_rows, ref_files: dict) -> str:
    pred_run = np.array(pred_rows).reshape(-1, 960)[:, 0]
    parts = []
    for name, path in ref_files.items():
        if not Path(path).exists():
            continue
        ref_run  = pd.read_csv(path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree    = (pred_run == ref_run).sum()
        me_only  = ((pred_run == 1) & (ref_run == 0)).sum()
        ref_only = ((pred_run == 0) & (ref_run == 1)).sum()
        parts.append(f"{name}: 일치 {agree}/740 | 내only {me_only} | {name}only {ref_only}")
    return "    " + " || ".join(parts)


def main():
    # ── 데이터 로드 및 전처리
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    train_vecs = build_run_vectors(train_X, train_run_ids)   # (500, 52) — GMM용
    test_vecs  = build_run_vectors(test_X,  test_run_ids)    # (740, 52)

    ref_files = {
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
        "Mahal": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
        "LOF":   str(OUTPUT_DIR / "output_exp15(LOF).csv"),
    }

    print("=" * 70)
    print("앙상블: GMM-tied + KMeans-Mahal — alpha 그리드")
    print("=" * 70)

    # ── GMM (Exp21과 동일: n_components=5, tied, reg_covar=1e-6)
    print("\n[GMM 학습 중] n_components=5, tied ...")
    gmm = GaussianMixture(
        n_components=5, covariance_type="tied",
        reg_covar=1e-6, random_state=RANDOM_SEED, max_iter=300, n_init=5,
    )
    gmm.fit(train_vecs.values)
    gmm_train_sc = -gmm.score_samples(train_vecs.values)   # 높을수록 이상
    gmm_test_sc  = -gmm.score_samples(test_vecs.values)

    # ── KMeans-Mahal (Exp17과 동일: k=50, 행 단위)
    print("[Mahal 학습 중] k=50, 행 단위 ...")
    mahal = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mahal.fit(train_X)
    # -decision_function() = min_mahal_dist (높을수록 이상)
    mahal_train_row = pd.Series(-mahal.decision_function(train_X), index=train_X.index)
    mahal_test_row  = pd.Series(-mahal.decision_function(test_X),  index=test_X.index)
    mahal_train_sc  = mahal_train_row.groupby(train_run_ids.values).mean().values
    mahal_test_sc   = mahal_test_row.groupby(test_run_ids.values).mean().values

    # ── Z-score 정규화 (훈련 run 점수 기준)
    gmm_tr_n,   gmm_te_n   = z_normalize(gmm_train_sc,   gmm_test_sc)
    mahal_tr_n, mahal_te_n = z_normalize(mahal_train_sc, mahal_test_sc)

    print(f"\n  GMM   train: μ={gmm_train_sc.mean():.3f}  σ={gmm_train_sc.std():.3f}")
    print(f"  Mahal train: μ={mahal_train_sc.mean():.3f}  σ={mahal_train_sc.std():.3f}")

    print("\n" + "=" * 70)
    print("alpha 탐색 (α=GMM 비중, 1-α=Mahal 비중)")
    print("=" * 70)

    test_run_index = test_vecs.index
    best_sep, best_alpha = -np.inf, None

    for alpha in ALPHA_GRID:
        ens_train = alpha * gmm_tr_n + (1 - alpha) * mahal_tr_n
        ens_test  = alpha * gmm_te_n + (1 - alpha) * mahal_te_n

        sep = (ens_test.mean() - ens_train.mean()) / ens_train.std()
        pred_rows = predict_from_ensemble(ens_test, test_run_index, test_run_ids, test_X.index)

        tag = f"alpha{alpha}"
        save_submission(pred_rows, test_X.index, OUTPUT_DIR / f"output_ensemble_{tag}.csv")

        print(f"\n[α={alpha:.1f} | GMM:{alpha:.0%}  Mahal:{1-alpha:.0%}]  sep={sep:.4f}  pos={pred_rows.mean():.4f}")
        print(compare_refs(pred_rows, ref_files))

        if sep > best_sep:
            best_sep   = sep
            best_alpha = alpha

    print(f"\n→ sep 기준 최적 alpha = {best_alpha} (sep={best_sep:.4f})")


if __name__ == "__main__":
    main()
