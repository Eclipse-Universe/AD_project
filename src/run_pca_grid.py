"""PCA 기반 이상 탐지 — SPE와 T² 통계 (row-level → run aggregation).

원리:
  정상 데이터로 PCA 학습 → 정상 운전의 부분공간(센서 상관 구조) 정의.

  SPE (Squared Prediction Error) = 재구성 오차:
    x → PCA(k) → x̂,  SPE = ||x - x̂||²
    정상: 작음 (정상 부분공간 안에 있음)
    이상: 큼   (이상으로 상관 구조 붕괴 → 부분공간 밖으로 이탈)

  T² (Hotelling's T²) = 부분공간 내 마할라노비스 거리:
    T² = Σ_j (z_j² / λ_j)   (z_j: j번째 주성분 점수, λ_j: 고유값)
    정상 부분공간 안에 있어도 중심에서 극단적으로 멀면 큰 값.

TEP 적합성:
  정상 운전 중 센서 간 강한 상관(xmv_7↔xmeas_12 등 r≈1.0)이 저차원 구조 형성.
  이상 발생 시 상관 붕괴 → SPE 민감 반응.
  화학공정 모니터링(MSPC) 논문의 표준 벤치마크.

참고:
  Chiang et al. (2001) Fault Detection and Diagnosis in Industrial Systems
  Lee et al. (2004) Chemometrics and Intelligent Laboratory Systems

실행: cd src && python run_pca_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data_loader import load_test, load_train
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42
N_COMPONENTS_GRID = [5, 10, 15, 20, 30]


def compute_spe_t2(pca: PCA, X: np.ndarray):
    """SPE(재구성 오차)와 T² 통계를 행 단위로 계산."""
    scores = pca.transform(X)
    X_hat  = pca.inverse_transform(scores)
    spe = np.sum((X - X_hat) ** 2, axis=1)
    t2  = np.sum(scores ** 2 / pca.explained_variance_, axis=1)
    return spe, t2


def run_to_pred(row_scores: np.ndarray, run_ids: pd.Series, X_index):
    """행 단위 점수 → run 평균 집계 → 이진 예측.
    SPE/T² 모두 높을수록 이상이므로 상위 RUN_CONTAMINATION 분위로 임계값 설정.
    """
    score_s   = pd.Series(row_scores, index=X_index)
    run_sc    = score_s.groupby(run_ids.values).mean()
    threshold = np.quantile(run_sc.values, 1 - RUN_CONTAMINATION)
    pred_run  = (run_sc >= threshold).astype(int)
    pred_rows = run_ids.map(pred_run).values
    return pred_rows, run_sc


def sep_index(train_sc: np.ndarray, test_sc: np.ndarray) -> float:
    """분리 지수 (높을수록 이상이 가정). (test_mean - train_mean) / train_std."""
    return (test_sc.mean() - train_sc.mean()) / train_sc.std()


def compare_refs(pred_rows, ref_files):
    pred_run = np.array(pred_rows).reshape(-1, 960)[:, 0]
    parts = []
    for name, path in ref_files.items():
        if not Path(path).exists():
            continue
        ref_run = pd.read_csv(path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree    = (pred_run == ref_run).sum()
        me_only  = ((pred_run == 1) & (ref_run == 0)).sum()
        ref_only = ((pred_run == 0) & (ref_run == 1)).sum()
        parts.append(f"{name}: agree={agree}/740 | mine={me_only} | {name}={ref_only}")
    return "    " + " || ".join(parts)


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    ref_files = {
        "LOF":   str(OUTPUT_DIR / "output_exp15(LOF).csv"),
        "Mahal": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
    }

    print("=" * 70)
    print("PCA 이상 탐지 — SPE / T² 비교 (row-level → run aggregation)")
    print(f"  훈련 행: {len(train_X):,}  테스트 행: {len(test_X):,}  피처: {train_X.shape[1]}")
    print("=" * 70)

    results = {}

    for k in N_COMPONENTS_GRID:
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        pca.fit(train_X.values)
        var_ratio = pca.explained_variance_ratio_.sum()

        tr_spe, tr_t2 = compute_spe_t2(pca, train_X.values)
        te_spe, te_t2 = compute_spe_t2(pca, test_X.values)

        pred_spe, run_spe = run_to_pred(te_spe, test_run_ids, test_X.index)
        pred_t2,  run_t2  = run_to_pred(te_t2,  test_run_ids, test_X.index)

        tr_run_spe = pd.Series(tr_spe, index=train_X.index).groupby(train_run_ids.values).mean()
        tr_run_t2  = pd.Series(tr_t2,  index=train_X.index).groupby(train_run_ids.values).mean()

        spe_row_sep = sep_index(tr_spe, te_spe)
        t2_row_sep  = sep_index(tr_t2,  te_t2)
        spe_run_sep = sep_index(tr_run_spe.values, run_spe.values)
        t2_run_sep  = sep_index(tr_run_t2.values,  run_t2.values)

        print(f"\n[k={k:2d}, 설명분산={var_ratio:.3f}]")
        print(f"  SPE  row sep={spe_row_sep:6.2f}  run sep={spe_run_sep:6.2f}  pos={pred_spe.mean():.4f}")
        print(compare_refs(pred_spe, ref_files))
        print(f"  T²   row sep={t2_row_sep:6.2f}  run sep={t2_run_sep:6.2f}  pos={pred_t2.mean():.4f}")
        print(compare_refs(pred_t2, ref_files))

        save_submission(pred_spe, test_X.index, OUTPUT_DIR / f"output_pca_spe_k{k}.csv")
        save_submission(pred_t2,  test_X.index, OUTPUT_DIR / f"output_pca_t2_k{k}.csv")

        results[k] = {
            "var_ratio": var_ratio,
            "spe_run_sep": spe_run_sep,
            "t2_run_sep":  t2_run_sep,
            "pred_spe": pred_spe,
            "pred_t2":  pred_t2,
        }

    best_k = max(results, key=lambda k: results[k]["spe_run_sep"])
    print(f"\n→ SPE run sep 기준 최적 k = {best_k} ({results[best_k]['var_ratio']:.3f} 설명분산)")


if __name__ == "__main__":
    main()
