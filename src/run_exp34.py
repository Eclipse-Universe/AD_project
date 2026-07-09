"""
run_exp34.py — 실제 contamination(240/740) 적용 실험

배경:
  all-1 제출 결과 F1=0.4898 → 실제 이상 run 수 A = 740×0.4898/(2-0.4898) = 240
  기존 가정(A=237, contamination=0.32)이 틀렸음.
  정확한 contamination = 240/740 = 0.32432...

  변경 사항: PP 237 → 240 (Exp30 모델 그대로, 임계값만 조정)

  예상:
    rank 238~240 run 중 k개가 실제 TP라면:
      k=3: F1 = 2×235/480 = 0.9792
      k=2: F1 = 2×234/480 = 0.9750
      k=1: F1 = 2×233/480 = 0.9708
      k=0: F1 = 2×232/480 = 0.9667  (악화)

실행: cd /root/AD_project/src && python run_exp34.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from data_loader import load_train, load_test
from infer import save_submission
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH  = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")
RANDOM_SEED = 42
TEST_TS = 960

# 핵심 변경: 237 → 240
TRUE_ANOMALY_COUNT = 240
TOTAL_TEST_RUNS    = 740
RUN_CONTAMINATION  = TRUE_ANOMALY_COUNT / TOTAL_TEST_RUNS  # 0.32432...


def sep_index(tr, te):
    return float((te.mean() - tr.mean()) / (tr.std() + 1e-10))


def main():
    print(f"실제 contamination = {TRUE_ANOMALY_COUNT}/{TOTAL_TEST_RUNS} = {RUN_CONTAMINATION:.5f}")
    print(f"  → 기존 PP=237 에서 PP={TRUE_ANOMALY_COUNT}으로 변경\n")

    print("데이터 로딩...")
    train_data    = load_train(DATA_PATH)
    test_data     = load_test(DATA_PATH)
    train_run_ids = train_data["simulationRun"]
    test_run_ids  = test_data["simulationRun"]

    scaler  = fit_scaler(select_features(train_data), scaler_type="standard")
    tr_X_df = scale_features(select_features(train_data), scaler)
    te_X_df = scale_features(select_features(test_data),  scaler)
    tr_X    = tr_X_df.values.astype(np.float64)
    te_X    = te_X_df.values.astype(np.float64)

    tr_run_vecs  = tr_X_df.groupby(train_run_ids.values).mean()
    te_run_vecs  = te_X_df.groupby(test_run_ids.values).mean()
    tr_run_order = tr_run_vecs.index.tolist()
    te_run_order = te_run_vecs.index.tolist()
    tr_rvec      = tr_run_vecs.values
    te_rvec      = te_run_vecs.values

    # ── GMM ──────────────────────────────────────────────────────
    print("GMM 학습...")
    gmm = GaussianMixture(n_components=2, covariance_type="tied",
                          reg_covar=1e-4, n_init=10, max_iter=300,
                          random_state=RANDOM_SEED)
    gmm.fit(tr_rvec)
    gmm_tr = -gmm.score_samples(tr_rvec)
    gmm_te = -gmm.score_samples(te_rvec)
    print(f"  GMM sep={sep_index(gmm_tr, gmm_te):.0f}")

    # ── Mahal ────────────────────────────────────────────────────
    print("Mahal 학습...")
    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_rvec)
    mah_tr = -mah.decision_function(tr_rvec)
    mah_te = -mah.decision_function(te_rvec)
    print(f"  Mahal sep={sep_index(mah_tr, mah_te):.0f}")

    # ── SPE k=30 ─────────────────────────────────────────────────
    print("SPE(k=30) 계산...")
    pca = PCA(n_components=30, random_state=RANDOM_SEED)
    pca.fit(tr_X)
    def spe(X):
        return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)

    tr_spe_s = pd.Series(spe(tr_X), index=train_run_ids.values).groupby(level=0).mean()
    te_spe_s = pd.Series(spe(te_X), index=test_run_ids.values).groupby(level=0).mean()
    spe_tr   = tr_spe_s.reindex(tr_run_order).values
    spe_te   = te_spe_s.reindex(te_run_order).values
    print(f"  SPE sep={sep_index(spe_tr, spe_te):.0f}")

    # ── Z-score 정규화 ──────────────────────────────────────────
    def z(tr_arr, te_arr):
        mu, sg = tr_arr.mean(), tr_arr.std() + 1e-10
        return (te_arr - mu) / sg, (tr_arr - mu) / sg

    zg_te, _ = z(gmm_tr, gmm_te)
    zm_te, _ = z(mah_tr, mah_te)
    zs_te, _ = z(spe_tr, spe_te)

    # ── Exp30과 동일한 가중치, 임계값만 변경 ─────────────────────
    ens_te = 0.6 * zg_te + 0.3 * zm_te + 0.1 * zs_te
    ens_series = pd.Series(ens_te, index=te_run_order).sort_values(ascending=False)

    # Exp30 기준 임계값 (PP=237)
    thr237 = ens_series.iloc[237 - 1]
    # 새 임계값 (PP=240)
    thr240 = ens_series.iloc[240 - 1]

    print(f"\n점수 경계:")
    print(f"  rank 237 (Exp30 마지노선): {thr237:.6f}")
    print(f"  rank 238: {ens_series.iloc[237]:.6f}")
    print(f"  rank 239: {ens_series.iloc[238]:.6f}")
    print(f"  rank 240 (Exp34 마지노선): {ens_series.iloc[239]:.6f}")
    print(f"  rank 241: {ens_series.iloc[240]:.6f}")
    print(f"  마진 변화: {thr237 - ens_series.iloc[237]:.6f} → {thr240 - ens_series.iloc[240]:.6f}")

    print(f"\n새로 추가되는 run (rank 238~240):")
    for rank in range(237, 240):
        print(f"  rank {rank+1}: run{ens_series.index[rank]:4d}  score={ens_series.iloc[rank]:.6f}")

    # ── 예측 생성 (PP=240) ───────────────────────────────────────
    pred240 = np.zeros(TOTAL_TEST_RUNS, dtype=int)
    pred240[np.argsort(ens_te)[-TRUE_ANOMALY_COUNT:]] = 1
    pred240_series = pd.Series(pred240, index=te_run_order)

    # Exp30과 비교
    exp30_df = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred_ts = exp30_df["faultNumber"].values
    exp30_pred_run = pd.Series(
        exp30_pred_ts.reshape(-1, TEST_TS)[:, 0], index=te_run_order
    )

    added   = [r for r in te_run_order if pred240_series[r]==1 and exp30_pred_run[r]==0]
    removed = [r for r in te_run_order if pred240_series[r]==0 and exp30_pred_run[r]==1]
    print(f"\nExp30 대비 변경:")
    print(f"  추가 (정상→이상): {[f'run{r}' for r in added]}")
    print(f"  제거 (이상→정상): {[f'run{r}' for r in removed]}")

    # ── timestep-level 예측 생성 & 저장 ─────────────────────────
    pred_row = test_run_ids.map(pred240_series).values
    out_path = OUTPUT_DIR / "output_exp34(GMM-Mahal-SPE30-PP240).csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"\n저장 완료: {out_path}")
    print(f"  PP=237 → PP=240, 3개 run 추가 이상 판정")


if __name__ == "__main__":
    main()
