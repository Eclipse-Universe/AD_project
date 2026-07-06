"""OC-SVM run-level 이상 탐지 — nu / gamma 그리드 탐색.

이전 시도와의 차이:
  Exp 5~7: 행 단위(250K행) + SGDOneClassSVM(선형 커널) → F1=0.38 실패
  이번:    run 평균 벡터(500×52) + OneClassSVM(RBF 커널) → 구조적으로 다른 접근

OC-SVM 이상 점수 방향:
  decision_function(x): 높을수록 정상(경계 안), 낮을수록 이상(경계 밖).
  → 하위 RUN_CONTAMINATION 분위 이하를 이상 run으로 판정 (GMM과 동일 방향).

그리드:
  nu    ∈ {0.05, 0.1, 0.2, 0.3}    — 훈련 오탐율 상한
  gamma ∈ {'scale', 'auto', 0.1}   — RBF 커널 폭

실행: cd src && python run_svm_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from data_loader import load_test, load_train
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

NU_GRID    = [0.05, 0.1, 0.2, 0.3]
GAMMA_GRID = ["scale", "auto", 0.1]


def build_run_vectors(X: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    return X.groupby(run_ids.values).mean()


def predict_runs(svm: OneClassSVM, test_vecs: pd.DataFrame,
                 test_run_ids: pd.Series, test_X_index):
    """decision_function → 하위 32%를 이상 run으로 판정 → 행 단위 예측 반환."""
    df_scores = svm.decision_function(test_vecs.values)   # 낮을수록 이상
    threshold  = np.quantile(df_scores, RUN_CONTAMINATION)
    pred_run   = pd.Series((df_scores <= threshold).astype(int), index=test_vecs.index)
    pred_rows  = test_run_ids.map(pred_run).values
    return pred_rows, df_scores


def sep_index(train_sc: np.ndarray, test_sc: np.ndarray) -> float:
    """(train_mean - test_mean) / train_std: 정상이 이상보다 decision_function이 높을수록 큼."""
    return (train_sc.mean() - test_sc.mean()) / train_sc.std()


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
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    train_vecs = build_run_vectors(train_X, train_run_ids)   # (500, 52)
    test_vecs  = build_run_vectors(test_X,  test_run_ids)    # (740, 52)

    ref_files = {
        "LOF":   str(OUTPUT_DIR / "output_exp15(LOF).csv"),
        "Mahal": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
    }

    print("=" * 70)
    print("OC-SVM run-level 이상 탐지 — nu × gamma 그리드")
    print(f"  훈련 run: {len(train_vecs)}  테스트 run: {len(test_vecs)}  피처: {train_vecs.shape[1]}")
    print("=" * 70)

    best_sep, best_cfg = -np.inf, None

    for nu in NU_GRID:
        for gamma in GAMMA_GRID:
            svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            svm.fit(train_vecs.values)

            train_df = svm.decision_function(train_vecs.values)
            pred_rows, test_df = predict_runs(svm, test_vecs, test_run_ids, test_X.index)

            sep = sep_index(train_df, test_df)
            pos = pred_rows.mean()

            tag = f"nu{nu}_g{gamma}"
            save_submission(pred_rows, test_X.index, OUTPUT_DIR / f"output_svm_{tag}.csv")

            print(f"\n[nu={nu}, gamma={gamma}]  sep={sep:.3f}  pos={pos:.4f}")
            print(compare_refs(pred_rows, ref_files))

            if sep > best_sep:
                best_sep = sep
                best_cfg = (nu, gamma, pred_rows, tag)

    nu_b, gamma_b, pred_b, tag_b = best_cfg
    print(f"\n→ 최적 설정: nu={nu_b}, gamma={gamma_b}  (sep={best_sep:.3f})")


if __name__ == "__main__":
    main()
