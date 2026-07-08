"""AE run-level 이상 탐지 — 아키텍처 그리드.

배경:
  기존 AE(Exp13, F1=0.9205)는 행 단위(250K×52) 학습 후 run 집계.
  구조적 발견: run-level 직접 모델링 > row-level 집계.
  → AE를 run 벡터(500×52)에 직접 학습하는 것을 처음 시도.

아키텍처 선택 근거 (run 벡터 PCA 분석 결과):
  PC1 단독 43%, PC1+2 = 66%, 90% 분산 = k13, 95% = k20.
  run 벡터는 타임스텝 평균화로 내재 차원수가 행 단위보다 훨씬 낮음.
  bottleneck=8  → 85.2% 분산만 설명, 정상 복원도 흐려져 부적합
  bottleneck=12 → 89.7% 분산 설명 (합리적 하한)
  bottleneck=16 → 93.1% 분산 설명 (권장)
  bottleneck=20 → 95.4% 분산 설명 (여유)

비교 아키텍처 (인코더 기준):
  A. 52 → 32 → 16  (bottleneck=16, 파라미터 ~4,352)
  B. 52 → 24 → 12  (bottleneck=12, 파라미터 ~3,072)
  C. 52 → 32 → 20  (bottleneck=20, 파라미터 ~4,608)
  D. 52 → 32 → 16 → 8 (기존 기본값, bottleneck=8 — 비교용)

500 훈련 샘플 기준 파라미터/샘플 비율: A=8.7, B=6.1, C=9.2, D=9.5

실행: cd src && python run_ae_runlevel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import load_test, load_train
from infer import save_submission
from model import TEPAutoencoder, train_autoencoder
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED = 42

# 비교할 아키텍처 (인코더 hidden_dims — 마지막이 bottleneck)
ARCH_GRID = [
    ([32, 16],    "bn16"),   # 권장 — bottleneck=16, 93% 분산
    ([24, 12],    "bn12"),   # 보수적 — bottleneck=12, 90% 분산
    ([32, 20],    "bn20"),   # 여유 — bottleneck=20, 95% 분산
    ([32, 16, 8], "bn8"),    # 기존 기본값 (비교 기준)
]


def build_run_vectors(X, run_ids):
    return X.groupby(run_ids.values).mean()


def sep_index(tr_sc, te_sc):
    return (te_sc.mean() - tr_sc.mean()) / tr_sc.std()


def predict_runs(te_sc, run_index, test_run_ids):
    thr = np.quantile(te_sc, 1 - RUN_CONTAMINATION)
    pred_run = pd.Series((te_sc >= thr).astype(int), index=run_index)
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

    scaler   = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X  = scale_features(select_features(train_data), scaler)
    test_X   = scale_features(select_features(test_data),  scaler)
    train_vecs = build_run_vectors(train_X, train_run_ids)  # 500 × 52
    test_vecs  = build_run_vectors(test_X,  test_run_ids)   # 740 × 52

    ref_files = {
        "Ens25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
        "Mahal": str(OUTPUT_DIR / "output_exp17(KMeans-Mahal).csv"),
    }

    print("=" * 65)
    print("AE run-level 아키텍처 그리드")
    print(f"  훈련: {train_vecs.shape}  테스트: {test_vecs.shape}")
    print("=" * 65)

    best_sep, best_tag = -np.inf, None

    for hidden_dims, tag in ARCH_GRID:
        bn = hidden_dims[-1]
        n_params = sum(
            hidden_dims[i-1] * hidden_dims[i]
            for i in range(1, len(hidden_dims))
        ) * 2 + 52 * hidden_dims[0] + hidden_dims[-1] * 52  # encoder + decoder 근사

        print(f"\n[아키텍처 52→{'→'.join(map(str, hidden_dims))}  bottleneck={bn}  params≈{n_params:,}]")

        torch.manual_seed(RANDOM_SEED)
        ae = train_autoencoder(
            train_vecs,
            hidden_dims=hidden_dims,
            epochs=300,
            batch_size=64,        # 500샘플 대비 적절
            lr=5e-4,
            val_ratio=0.1,
            patience=20,
            random_state=RANDOM_SEED,
        )
        ae.eval()

        tr_t = torch.tensor(train_vecs.values.astype("float32"))
        te_t = torch.tensor(test_vecs.values.astype("float32"))
        tr_sc = ae.reconstruction_error(tr_t).numpy()
        te_sc = ae.reconstruction_error(te_t).numpy()

        sep  = sep_index(tr_sc, te_sc)
        pred = predict_runs(te_sc, test_vecs.index, test_run_ids)
        pos  = pred.mean()

        tag_full = f"ae_run_{tag}"
        save_submission(pred, test_X.index, OUTPUT_DIR / f"output_{tag_full}.csv")

        print(f"  sep={sep:.3f}  pos={pos:.4f}")
        print(f"  train RE: μ={tr_sc.mean():.6f}  σ={tr_sc.std():.6f}")
        print(f"  test  RE: μ={te_sc.mean():.6f}  σ={te_sc.std():.6f}")
        print(compare(pred, ref_files))

        if sep > best_sep:
            best_sep = sep
            best_tag = (hidden_dims, tag)

    print(f"\n→ sep 기준 최적: 52→{'→'.join(map(str, best_tag[0]))}  sep={best_sep:.3f}")


if __name__ == "__main__":
    main()
