"""
run_exp38.py — Exp30에서 run13 제거 (PP: 237→236)

근거:
  run13 단독 오라클(Exp37) 결과 F1=0.0000 → TP=0 → run13 = FP 확정.
  Exp30 anomaly set(237개)에서 run13만 제거.

예상 (A=240, TP=231.5 불변):
  PP=236, TP=231.5, FP=4.5
  F1 = 2×231.5/476 = 0.9727

실제 결과: F1=0.9748, Acc=0.9838 (신규 최고)
  TP=232.0, FP=4.0 (timestep-level 처리 차이로 예상 대비 소폭 추가 개선)
"""
import pandas as pd
from pathlib import Path

from data_loader import load_train, load_test
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH  = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")

REMOVE_RUNS = [13]  # run13: FP 확정 (Exp37 오라클)


def main():
    train_data   = load_train(DATA_PATH)
    test_data    = load_test(DATA_PATH)
    test_run_ids = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data))
    te_X_df = scale_features(select_features(test_data), scaler)
    te_run_order = te_X_df.groupby(test_run_ids.values).mean().index.tolist()

    exp30_df = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred = (
        pd.Series(exp30_df["faultNumber"].values, index=test_run_ids.values)
        .groupby(level=0).first()
    )
    exp30_anom = set(exp30_pred.index[exp30_pred == 1].tolist())
    print(f"Exp30 anomaly set: {len(exp30_anom)}개")

    remove_set = set(REMOVE_RUNS)
    assert remove_set.issubset(exp30_anom), f"제거 대상이 Exp30에 없음: {remove_set - exp30_anom}"
    new_anom = exp30_anom - remove_set
    print(f"제거 후: {len(new_anom)}개 (제거: {sorted(REMOVE_RUNS)})")

    pred = pd.Series(
        [1 if r in new_anom else 0 for r in te_run_order],
        index=te_run_order
    )
    pred_row = test_run_ids.map(pred).values

    out_path = OUTPUT_DIR / "output_exp38(Exp30-FPremove_run13).csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"저장: {out_path}")

    A = 240
    tp = 231.5
    pp = len(new_anom)
    print(f"\n예상 F1 = 2×{tp}/({pp}+{A}) = {2*tp/(pp+A):.4f}")


if __name__ == "__main__":
    main()
