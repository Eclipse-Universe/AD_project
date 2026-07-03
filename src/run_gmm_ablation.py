"""GMM 과적합 완화 실험 — reg_covar / covariance_type / n_components 탐색.

목적:
  Exp 20 (GMM k=5 full, F1=0.9367)의 구조적 과적합 위험 완화.
  성분당 파라미터(1,430) >> 데이터(100)로 중간 평가에 과적합됐을 가능성.

실험 설계:
  A. reg_covar 그리드 (k=5, full): {1e-6, 1e-3, 1e-2, 1e-1, 1.0}
     - Σ_k += reg_covar*I → 등방성 방향으로 수축 → 일반화 개선
  B. covariance_type 비교 (k=5, reg_covar=1e-6): full / tied / diag
     - tied: 파라미터 1,638 (full 7,150 대비 -78%)
     - diag: 파라미터 520 (상관 구조 포기, 안정성 최대)
  C. 소수 성분 (k=2,3, full): 성분당 데이터 250/167 → 공분산 추정 안정화

판단 기준: LOF-A / KMeans-Mahal과 일치도 (로컬 검증 지표)

실행: cd src && python run_gmm_ablation.py
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


def build_run_vectors(X, run_ids):
    return X.groupby(run_ids.values).mean()


def run_gmm(train_vecs, test_vecs, test_run_ids, test_X_index,
            n_components, covariance_type, reg_covar, name):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=RANDOM_SEED,
        max_iter=300,
        n_init=5,
    )
    gmm.fit(train_vecs.values)

    train_sc = gmm.score_samples(train_vecs.values)
    test_sc  = gmm.score_samples(test_vecs.values)

    sep = (train_sc.mean() - test_sc.mean()) / train_sc.std()

    threshold     = np.quantile(test_sc, RUN_CONTAMINATION)
    anomalous     = test_vecs.index[test_sc <= threshold]
    pred_run      = pd.Series(0, index=test_vecs.index)
    pred_run[anomalous] = 1
    pred_rows = test_run_ids.map(pred_run).values

    save_submission(pred_rows, test_X_index,
                    OUTPUT_DIR / f"output_gmm_ablation_{name}.csv")
    return pred_rows, train_sc, test_sc, sep


def compare(pred_rows, label, ref_files):
    pred_run = np.array(pred_rows).reshape(-1, 960)[:, 0]
    parts = []
    for ref_name, ref_path in ref_files.items():
        if not Path(ref_path).exists():
            continue
        ref_run = pd.read_csv(ref_path)["faultNumber"].values.reshape(-1, 960)[:, 0]
        agree    = (pred_run == ref_run).sum()
        only_me  = ((pred_run == 1) & (ref_run == 0)).sum()
        only_ref = ((pred_run == 0) & (ref_run == 1)).sum()
        parts.append(f"{ref_name}: 일치 {agree}/740 | 내only {only_me} | {ref_name}only {only_ref}")
    return "  " + " || ".join(parts)


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
        "LOF":   str(OUTPUT_DIR / "output_exp15(LOF).csv"),
        "Mahal": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
        "GMM-k5(Exp20)": str(OUTPUT_DIR / "output_exp20(GMM).csv"),
    }

    print("=" * 70)
    print("A. reg_covar 그리드 (k=5, full covariance)")
    print("=" * 70)
    reg_results = {}
    for reg in [1e-6, 1e-3, 1e-2, 1e-1, 1.0]:
        name = f"reg{reg}"
        pred, tr_sc, te_sc, sep = run_gmm(
            train_vecs, test_vecs, test_run_ids, test_X.index,
            n_components=5, covariance_type="full", reg_covar=reg, name=name,
        )
        pos_rate = pred.mean()
        print(f"\n[reg_covar={reg}]  sep={sep:.1f}  pos_rate={pos_rate:.4f}")
        print(compare(pred, name, ref_files))
        reg_results[reg] = (pred, sep)

    print("\n" + "=" * 70)
    print("B. covariance_type 비교 (k=5, reg_covar=1e-6)")
    print("=" * 70)
    for cov_type in ["full", "tied", "diag"]:
        params = {"full": 5*1430, "tied": 5*52+1378, "diag": 5*104}[cov_type]
        pred, tr_sc, te_sc, sep = run_gmm(
            train_vecs, test_vecs, test_run_ids, test_X.index,
            n_components=5, covariance_type=cov_type, reg_covar=1e-6, name=f"cov_{cov_type}",
        )
        pos_rate = pred.mean()
        print(f"\n[type={cov_type}, params≈{params}]  sep={sep:.1f}  pos_rate={pos_rate:.4f}")
        print(compare(pred, cov_type, ref_files))

    print("\n" + "=" * 70)
    print("C. 소수 성분 (k=2,3, full, reg_covar=1e-6)")
    print("=" * 70)
    for k in [2, 3]:
        data_per_comp = 500 // k
        pred, tr_sc, te_sc, sep = run_gmm(
            train_vecs, test_vecs, test_run_ids, test_X.index,
            n_components=k, covariance_type="full", reg_covar=1e-6, name=f"k{k}",
        )
        pos_rate = pred.mean()
        print(f"\n[k={k}, 성분당 데이터≈{data_per_comp}]  sep={sep:.1f}  pos_rate={pos_rate:.4f}")
        print(compare(pred, f"k{k}", ref_files))


if __name__ == "__main__":
    main()
