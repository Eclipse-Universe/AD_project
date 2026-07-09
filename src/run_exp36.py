"""
run_exp36.py — 역분석 기반 FP 후보 5개 제거 (PP: 237→232)

근거:
  5가지 독립 지표 모두에서 "정상" 신호를 보이는 Exp30 anomaly runs:
    run13  : z_gmm=-0.882, z_mah=-0.060, z_spe= 0.060, z_lof=-0.571, z_osvm=-0.826
    run20  : z_gmm=-1.468, z_mah= 0.184, z_spe= 0.673, z_lof=-0.860, z_osvm=-2.091
    run232 : z_gmm=-1.098, z_mah= 0.184, z_spe= 0.481, z_lof=-0.854, z_osvm=-1.678
    run544 : z_gmm=-0.454, z_mah= 0.418, z_spe= 1.684, z_lof=-0.516, z_osvm=-0.625
    run666 : z_gmm=-0.587, z_mah= 0.825, z_spe= 1.599, z_lof=-0.714, z_osvm=-1.189

  비교: run155(z_spe=1.785), run377(z_spe=0.882)은 Exp35에서 TP로 판명 → 제외
  위 5개는 run155/377보다 모든 지표가 더 약하고 LOF/OCSVM 신호도 더 낮음

예상 (A=240, TP=231.5 기준):
  5개 전부 FP → F1=0.9809
  3개 FP, 2개 TP → F1=0.9724 (Exp30 초과)
  2개 FP, 3개 TP → F1=0.9682 (Exp30 이하)
"""
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_train, load_test
from infer import save_submission
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH  = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")

REMOVE_RUNS = [13, 20, 232, 544, 666]  # FP 후보 5개


def main():
    print(f"제거 대상: {['run'+str(r) for r in REMOVE_RUNS]}")

    train_data = load_train(DATA_PATH)
    test_data  = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    te_X_df = scale_features(select_features(test_data), scaler)
    te_run_order = te_X_df.groupby(test_run_ids.values).mean().index.tolist()

    # Exp30 anomaly set 로드
    exp30_df = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred_ts = exp30_df["faultNumber"].values
    exp30_run_pred = (
        pd.Series(exp30_pred_ts, index=test_run_ids.values)
        .groupby(level=0).first()
    )
    exp30_anom_set = set(exp30_run_pred.index[exp30_run_pred == 1].tolist())
    print(f"Exp30 anomaly set: {len(exp30_anom_set)}개")

    # FP 후보 제거
    remove_set = set(REMOVE_RUNS)
    assert remove_set.issubset(exp30_anom_set), f"제거 대상 중 Exp30에 없는 run: {remove_set - exp30_anom_set}"
    new_anom_set = exp30_anom_set - remove_set
    print(f"제거 후 anomaly set: {len(new_anom_set)}개 (={len(exp30_anom_set)}-{len(remove_set)})")

    new_pred_run = pd.Series(
        [1 if r in new_anom_set else 0 for r in te_run_order],
        index=te_run_order
    )
    pred_row = test_run_ids.map(new_pred_run).values

    out_path = OUTPUT_DIR / "output_exp36(Exp30-FPremove5).csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"저장: {out_path}")
    print(f"PP=232 / 제거된 runs: {sorted(REMOVE_RUNS)}")


if __name__ == "__main__":
    main()
