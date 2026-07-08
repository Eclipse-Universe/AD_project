"""AE row-level v2 — expand-first 아키텍처 (멘토 조언 적용).

Exp13(F1=0.9205) 개선 시도.

Exp13 문제:
  아키텍처 52→32→16→8: 첫 hidden layer(32)가 input(52)보다 작음.
  첫 레이어부터 압축하면 피처 간 비선형 관계를 학습하기 전에 정보 손실.

멘토 조언 (올바른 적용 맥락: row-level, 250K 샘플):
  52 → 128 → 64 → 32 → bottleneck
  첫 레이어를 input보다 크게(expand) → 충분한 표현 학습 후 압축.
  250K 샘플에서는 33K params → 0.13x 비율, 과적합 위험 없음.

비교 아키텍처:
  A. 52→128→64→32   (bn=32, Exp13 대비 7.3x params, 가장 자연스러운 확장)
  B. 52→128→64→16   (bn=16, 더 강한 압축)
  C. 52→128→32      (층 줄이고 expansion 유지)
  D. 52→64→32       (중간 단계)
  E. 52→32→16→8     (Exp13 기준선, 비교용)

실행: cd src && python run_ae_rowlevel_v2.py
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

ARCH_GRID = [
    ([32, 16, 8],   "Exp13_base"),   # 기준선
    ([128, 64, 32], "h128_64_bn32"), # 권장 — expand-first
    ([128, 64, 16], "h128_64_bn16"), # bn 더 좁게
    ([128, 32],     "h128_bn32"),    # 2층 expand
    ([64, 32],      "h64_bn32"),     # 소폭 expand
]


def sep_index(tr_sc, te_sc):
    return (te_sc.mean() - tr_sc.mean()) / tr_sc.std()


def predict_runs(score_series, run_ids):
    run_sc = score_series.groupby(run_ids.values).mean()
    thr = np.quantile(run_sc.values, 1 - RUN_CONTAMINATION)
    pred_run = pd.Series((run_sc >= thr).astype(int), index=run_sc.index)
    return run_ids.map(pred_run).values, run_sc


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
    return "  ".join(parts)


def main():
    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    train_X = scale_features(select_features(train_data), scaler)
    test_X  = scale_features(select_features(test_data),  scaler)

    ref_files = {
        "Ens25": str(OUTPUT_DIR / "output_exp25(Ensemble3-LOF10).csv"),
        "GMM":   str(OUTPUT_DIR / "output_exp21(GMM-tied).csv"),
        "AEv1":  str(OUTPUT_DIR / "output_exp13(AE).csv"),
    }

    print("=" * 70)
    print("AE row-level v2 — expand-first 아키텍처 그리드")
    print(f"  train: {train_X.shape}  test: {test_X.shape}")
    print("=" * 70)

    results = []

    for hidden_dims, tag in ARCH_GRID:
        dims = [52] + hidden_dims
        enc_params = sum(dims[i] * dims[i+1] for i in range(len(dims)-1))
        n_params = enc_params * 2
        ratio = n_params / len(train_X)
        arch_str = "52→" + "→".join(map(str, hidden_dims))
        bn = hidden_dims[-1]

        print(f"\n[{arch_str}  bn={bn}  params={n_params:,}  ratio={ratio:.4f}x]")

        ae = train_autoencoder(
            train_X,
            hidden_dims=hidden_dims,
            epochs=100,
            batch_size=256,
            lr=1e-3,
            val_ratio=0.1,
            patience=10,
            random_state=RANDOM_SEED,
            weight_decay=0.0,
        )
        ae.eval()

        # 행 단위 재구성 오차 계산
        X_np = train_X.values.astype("float32")
        X_te = test_X.values.astype("float32")
        chunk = 10_000

        tr_re = []
        for i in range(0, len(X_np), chunk):
            t = torch.tensor(X_np[i:i+chunk])
            with torch.no_grad():
                tr_re.append(ae.reconstruction_error(t).numpy())
        tr_re = np.concatenate(tr_re)

        te_re = []
        for i in range(0, len(X_te), chunk):
            t = torch.tensor(X_te[i:i+chunk])
            with torch.no_grad():
                te_re.append(ae.reconstruction_error(t).numpy())
        te_re = np.concatenate(te_re)

        tr_series = pd.Series(tr_re, index=train_X.index)
        te_series = pd.Series(te_re, index=test_X.index)

        pred, run_sc_te = predict_runs(te_series, test_run_ids)
        tr_run_sc = tr_series.groupby(train_run_ids.values).mean().values

        sep = sep_index(tr_run_sc, run_sc_te.values)

        save_submission(pred, test_X.index,
                        OUTPUT_DIR / f"output_ae_row_{tag}.csv")

        print(f"  sep={sep:.2f}  pos={pred.mean():.4f}")
        print(f"  train RE: μ={tr_re.mean():.6f}  σ={tr_re.std():.6f}")
        print(f"  " + compare(pred, ref_files))

        results.append((sep, arch_str, bn, n_params))

    print("\n" + "=" * 70)
    print("비교 요약 (sep 내림차순)")
    print(f"  {'아키텍처':<25} {'bn':>4} {'params':>8} {'sep':>12}")
    for sep, arch, bn, p in sorted(results, reverse=True):
        print(f"  {arch:<25} {bn:>4} {p:>8,} {sep:>12.1f}")


if __name__ == "__main__":
    main()
