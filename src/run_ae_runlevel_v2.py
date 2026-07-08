"""AE run-level — expand-then-compress 아키텍처 그리드.

멘토 피드백: hidden layer가 input(52)보다 작으면 안 된다.
첫 레이어부터 압축하는 구조(52→32→...)는 정상 run의 복잡한 관계를
학습하기 전에 정보를 버림 → expand-first 구조(52→H1→...→bn, H1>52)로 재설계.

기존 최적 (run_ae_runlevel.py):
  52→32→20 (bottleneck=20): sep=32,364  train RE=0.000930
  → 비교 기준선으로 유지

500 샘플 과적합 대응:
  파라미터/샘플 비율이 높을수록 weight_decay 강화.
  H1=64:   params≈9K~11K, ratio≈18~22x → weight_decay=1e-4
  H1=128:  params≈20K~27K, ratio≈40~55x → weight_decay=1e-3
  모든 실험: patience=25, batch=32 (배치 작게 → 정규화 효과), val_ratio=0.15

PCA 기반 bottleneck 근거 (run 벡터 500×52):
  90% 분산 도달 k=13 → bottleneck ≥ 16 적합
  95% 분산 도달 k=20 → bottleneck=20 여유

실행: cd src && python run_ae_runlevel_v2.py
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

# (hidden_dims, tag, weight_decay)
# 기준선: 52→32→20 (이전 최적, compressive)
# 신규:   첫 H1 > 52 (expand-first)
ARCH_GRID = [
    # ── 기준선 (비교용)
    ([32, 20],     "bn20_base",  0.0),

    # ── H1=64 expand (파라미터 적음, weight_decay 약하게)
    ([64, 20],     "h64_bn20",   1e-4),   # 52→64→20, params≈9.2K
    ([64, 32],     "h64_bn32",   1e-4),   # 52→64→32, params≈10.8K
    ([64, 16],     "h64_bn16",   1e-4),   # 52→64→16, params≈8.4K

    # ── H1=128 expand (파라미터 많음, weight_decay 강하게)
    ([128, 32],    "h128_bn32",  1e-3),   # 52→128→32, params≈21.5K
    ([128, 20],    "h128_bn20",  1e-3),   # 52→128→20, params≈19.2K

    # ── 3-layer (deep expand-compress)
    ([128, 64, 20], "h128_64_bn20", 1e-3),  # 52→128→64→20, params≈27.8K
    ([64,  32, 16], "h64_32_bn16",  1e-4),  # 52→64→32→16,  params≈12.2K
]


def build_run_vectors(X, run_ids):
    return X.groupby(run_ids.values).mean()


def sep_index(tr_sc, te_sc):
    return (te_sc.mean() - tr_sc.mean()) / tr_sc.std()


def count_params(input_dim, hidden_dims):
    dims = [input_dim] + hidden_dims
    enc = sum(dims[i] * dims[i+1] for i in range(len(dims)-1))
    dec = enc
    return enc + dec


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
        parts.append(f"{name}:{agree}/740(+{me_only}/-{ref_only})")
    return "  " + "  ".join(parts)


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler     = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X    = scale_features(select_features(train_data), scaler)
    test_X     = scale_features(select_features(test_data),  scaler)
    train_vecs = build_run_vectors(train_X, train_run_ids)   # 500 × 52
    test_vecs  = build_run_vectors(test_X,  test_run_ids)    # 740 × 52

    ref_files = {
        "Ens25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
    }

    print("=" * 70)
    print("AE run-level v2 — expand-first 아키텍처 그리드")
    print(f"  train: {train_vecs.shape}  test: {test_vecs.shape}")
    print("=" * 70)

    results = []

    for hidden_dims, tag, wd in ARCH_GRID:
        n_params = count_params(52, hidden_dims)
        ratio    = n_params / 500
        arch_str = "52→" + "→".join(map(str, hidden_dims))
        bn       = hidden_dims[-1]

        print(f"\n[{arch_str}  bn={bn}  params={n_params:,}  ratio={ratio:.1f}x  wd={wd}]")

        ae = train_autoencoder(
            train_vecs,
            hidden_dims=hidden_dims,
            epochs=500,
            batch_size=32,
            lr=3e-4,
            val_ratio=0.15,
            patience=25,
            random_state=RANDOM_SEED,
            weight_decay=wd,
        )
        ae.eval()

        tr_t  = torch.tensor(train_vecs.values.astype("float32"))
        te_t  = torch.tensor(test_vecs.values.astype("float32"))
        tr_sc = ae.reconstruction_error(tr_t).numpy()
        te_sc = ae.reconstruction_error(te_t).numpy()

        sep  = sep_index(tr_sc, te_sc)
        pred = predict_runs(te_sc, test_vecs.index, test_run_ids)

        save_submission(pred, test_X.index, OUTPUT_DIR / f"output_ae_run_{tag}.csv")

        print(f"  sep={sep:>10.2f}  trainRE: μ={tr_sc.mean():.6f} σ={tr_sc.std():.6f}")
        print(f"  testRE:  μ={te_sc.mean():.4f}  σ={te_sc.std():.4f}  pos={pred.mean():.4f}")
        print(compare(pred, ref_files))

        results.append((sep, arch_str, tag, bn, n_params, wd))

    print("\n" + "=" * 70)
    print("아키텍처 비교 요약 (sep 내림차순)")
    print(f"  {'아키텍처':<25} {'bn':>4} {'params':>8} {'ratio':>7} {'wd':>8} {'sep':>12}")
    print("  " + "-" * 68)
    for sep, arch, tag, bn, n_params, wd in sorted(results, reverse=True):
        ratio = n_params / 500
        print(f"  {arch:<25} {bn:>4} {n_params:>8,} {ratio:>6.1f}x {wd:>8.0e} {sep:>12.1f}")


if __name__ == "__main__":
    main()
