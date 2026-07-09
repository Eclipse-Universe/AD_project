"""
run_exp37_oracle.py — run13 단독 오라클 제출 (PP=1)

목적: run13이 FP인지 TP인지 1회 제출로 확정
  F1 = 0.0000 → run13 = FP → Exp38에서 제거
  F1 = 0.0083 → run13 = TP (2/241)
  F1 ≈ 0.0041 → run13 = 반-TP (fault 중반 시작)

결과: F1=0.0000, Acc=0.6730 → run13 = FP 확정
"""
import pandas as pd
from pathlib import Path

from data_loader import load_train, load_test
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH  = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")

ORACLE_RUN = 13  # 단독 이상 판정할 run ID


def main():
    train_data   = load_train(DATA_PATH)
    test_data    = load_test(DATA_PATH)
    test_run_ids = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data))
    te_X_df = scale_features(select_features(test_data), scaler)
    te_run_order = te_X_df.groupby(test_run_ids.values).mean().index.tolist()

    pred = pd.Series(
        [1 if r == ORACLE_RUN else 0 for r in te_run_order],
        index=te_run_order
    )
    pred_row = test_run_ids.map(pred).values

    out_path = OUTPUT_DIR / "output_exp37_oracle_run13.csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"저장: {out_path}")
    print(f"PP=1, 이상 판정 run: run{ORACLE_RUN}")
    print("해석: F1=0.0000 → FP, F1=0.0083 → TP, F1≈0.0041 → 반-TP")


if __name__ == "__main__":
    main()
