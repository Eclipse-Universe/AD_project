"""GMM 하이퍼파라미터 재탐색 — BIC/AIC 기반 통계적 모델 선택.

Exp21(n_components=5, tied, F1=0.9372) 이후 탐색 공백 구간 {2~9} 및
covariance_type='diag' 를 BIC/AIC와 함께 분석한다.

BIC/AIC를 쓰는 이유:
  - 리더보드 제출(ground truth) 없이도 통계적으로 최적 구조를 선택 가능
  - BIC: 파라미터 수 × log(n) 페널티 → 소규모 데이터(500 run)에서 과적합 억제
  - AIC: 파라미터 수 × 2 페널티 → BIC보다 복잡한 모델 허용
  - 두 기준이 일치하는 지점 = 가장 신뢰할 수 있는 최적 구조

covariance_type 비교:
  - 'full':  각 성분이 독립 Σ_k → 파라미터 k×(d + d(d+1)/2) → 500샘플에서 과적합
  - 'tied':  모든 성분 공유 Σ → 파라미터 k×d + d(d+1)/2 → 현재 (Exp21)
  - 'diag':  대각 Σ (피처 독립 가정) → 파라미터 k×2d → 가장 적은 파라미터
  TEP 데이터는 센서 상관이 높으므로 'diag'가 이론상 부적합하지만
  소규모 데이터에서의 분산-편향 트레이드오프 실험적으로 확인 필요.

실행: cd src && python run_gmm_grid_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from data_loader import load_test, load_train
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

N_COMPONENTS_GRID = [2, 3, 4, 5, 6, 7, 8, 10]
COV_TYPES = ["tied", "diag", "full"]

REF_FILES = {
    "Exp25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
    "Exp21": str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
}


def sep_index(tr_scores: np.ndarray, te_scores: np.ndarray) -> float:
    return (te_scores.mean() - tr_scores.mean()) / tr_scores.std()


def compare(pred_row: np.ndarray) -> str:
    """pred_row: row-level 예측 (원본 test 데이터 순서 유지). reshape로 run-level 비교."""
    pred_run = pred_row.reshape(-1, 960)[:, 0]
    parts = []
    for name, path in REF_FILES.items():
        if not Path(path).exists():
            continue
        ref = pd.read_csv(path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree   = (pred_run == ref).sum()
        me_only = ((pred_run == 1) & (ref == 0)).sum()
        r_only  = ((pred_run == 0) & (ref == 1)).sum()
        parts.append(f"{name}:{agree}/740(+{me_only}/-{r_only})")
    return "  ".join(parts)


def n_params(k: int, d: int, cov_type: str) -> int:
    means = k * d
    if cov_type == "full":
        cov = k * d * (d + 1) // 2
    elif cov_type == "tied":
        cov = d * (d + 1) // 2
    else:  # diag
        cov = k * d
    mixing = k - 1
    return means + cov + mixing


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler    = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X   = scale_features(select_features(train_data), scaler)
    test_X    = scale_features(select_features(test_data),  scaler)
    train_vecs = train_X.groupby(train_run_ids.values).mean()
    test_vecs  = test_X.groupby(test_run_ids.values).mean()

    X_tr = train_vecs.values   # (500, 52)
    X_te = test_vecs.values    # (740, 52)
    n, d = X_tr.shape
    run_idx = test_vecs.index

    print("=" * 72)
    print(f"GMM 재탐색: train={X_tr.shape}  test={X_te.shape}")
    print(f"파라미터 수 기준: n={n}  d={d}")
    print("=" * 72)

    results = []

    for cov_type in COV_TYPES:
        print(f"\n{'─'*72}")
        print(f"covariance_type = '{cov_type}'")
        print(f"{'─'*72}")
        print(f"  {'k':>3}  {'BIC':>10}  {'AIC':>10}  {'params':>8}  "
              f"{'sep':>10}  {'Exp25 agree':>13}  {'pos%':>6}")
        print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*8}  "
              f"{'─'*10}  {'─'*13}  {'─'*6}")

        for k in N_COMPONENTS_GRID:
            p = n_params(k, d, cov_type)
            # 'full'은 파라미터 수가 샘플보다 훨씬 많아 수렴 불안정 → reg_covar 높게
            reg = 1e-4 if cov_type == "full" else 1e-6

            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                reg_covar=reg,
                max_iter=300,
                n_init=10,
                random_state=RANDOM_SEED,
            )
            try:
                gmm.fit(X_tr)
            except Exception as e:
                print(f"  {k:>3}  FAILED: {e}")
                continue

            bic = gmm.bic(X_tr)
            aic = gmm.aic(X_tr)

            tr_sc = -gmm.score_samples(X_tr)
            te_sc = -gmm.score_samples(X_te)
            sep   = sep_index(tr_sc, te_sc)

            thr      = np.quantile(te_sc, 1 - RUN_CONTAMINATION)
            pred_run = (te_sc >= thr).astype(int)
            pred_row = test_run_ids.map(
                pd.Series(pred_run, index=run_idx)
            ).values
            pos = pred_run.mean()

            cmp = compare(pred_row)
            print(f"  {k:>3}  {bic:>10.1f}  {aic:>10.1f}  {p:>8,}  "
                  f"{sep:>10.1f}  {cmp}  {pos:>6.4f}")

            tag  = f"gmm_k{k}_{cov_type}"
            save_submission(pred_row, test_X.index, OUTPUT_DIR / f"output_{tag}.csv")
            results.append(dict(k=k, cov=cov_type, bic=bic, aic=aic,
                                params=p, sep=sep, pos=pos))

    # 요약 테이블 (BIC 오름차순)
    df = pd.DataFrame(results).sort_values("bic")
    print("\n" + "=" * 72)
    print("요약 — BIC 오름차순 (낮을수록 좋음)")
    print("=" * 72)
    print(df[["k", "cov", "bic", "aic", "params", "sep", "pos"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\n→ BIC 최적: k={int(best.k)}  cov='{best.cov}'  "
          f"BIC={best.bic:.1f}  sep={best.sep:.1f}")


if __name__ == "__main__":
    main()
