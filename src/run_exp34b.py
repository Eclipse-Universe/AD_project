"""
run_exp34b.py — 원래 Exp30 예측 보존 + 3개 추가 (PP237→240)

방식:
  - 원래 output_exp30 파일에서 237개 anomaly run을 그대로 유지
  - 재계산한 앙상블 점수에서 Exp30 정상 판정(503개) 중 점수 상위 3개를 추가
  → 총 240개 anomaly run

장점:
  - run155, run293, run377, run381 등 Exp30에서 확정된 TP들을 잃지 않음
  - 모델 비결정성 영향 없이 순수하게 3개 run만 추가

실행: cd /root/AD_project/src && python run_exp34b.py
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
TRUE_ANOMALY_COUNT = 240


def main():
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
    te_run_order = te_run_vecs.index.tolist()

    # ── 앙상블 점수 재계산 ──────────────────────────────────────
    gmm = GaussianMixture(n_components=2, covariance_type="tied",
                          reg_covar=1e-4, n_init=10, max_iter=300,
                          random_state=RANDOM_SEED)
    gmm.fit(tr_run_vecs.values)
    gmm_tr = -gmm.score_samples(tr_run_vecs.values)
    gmm_te = -gmm.score_samples(te_run_vecs.values)

    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=RANDOM_SEED)
    mah.fit(tr_run_vecs.values)
    mah_tr = -mah.decision_function(tr_run_vecs.values)
    mah_te = -mah.decision_function(te_run_vecs.values)

    pca = PCA(n_components=30, random_state=RANDOM_SEED)
    pca.fit(tr_X)
    def spe(X):
        return ((X - pca.inverse_transform(pca.transform(X)))**2).sum(axis=1)

    spe_tr = pd.Series(spe(tr_X), index=train_run_ids.values).groupby(level=0).mean().reindex(tr_run_vecs.index).values
    spe_te = pd.Series(spe(te_X), index=test_run_ids.values).groupby(level=0).mean().reindex(te_run_order).values

    def z(tr_arr, te_arr):
        mu, sg = tr_arr.mean(), tr_arr.std() + 1e-10
        return (te_arr - mu) / sg, (tr_arr - mu) / sg

    zg_te, _ = z(gmm_tr, gmm_te)
    zm_te, _ = z(mah_tr, mah_te)
    zs_te, _ = z(spe_tr, spe_te)

    ens_te  = 0.6 * zg_te + 0.3 * zm_te + 0.1 * zs_te
    ens_ser = pd.Series(ens_te, index=te_run_order).sort_values(ascending=False)

    # ── 원래 Exp30 anomaly set 로드 ─────────────────────────────
    exp30_df = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred_ts = exp30_df["faultNumber"].values
    exp30_run_pred = pd.Series(exp30_pred_ts, index=test_run_ids.values).groupby(level=0).first()

    exp30_anom_set = set(exp30_run_pred.index[exp30_run_pred == 1].tolist())
    exp30_norm_set = set(exp30_run_pred.index[exp30_run_pred == 0].tolist())
    print(f"Exp30 이상 run 수: {len(exp30_anom_set)} (보존)")
    print(f"Exp30 정상 run 수: {len(exp30_norm_set)}")

    # ── 정상 판정 run 중 앙상블 점수 상위 3개 찾기 ──────────────
    norm_scores = ens_ser[ens_ser.index.isin(exp30_norm_set)]
    top3_additions = norm_scores.head(3)

    print(f"\n추가할 3개 run (Exp30 정상 판정 중 앙상블 점수 상위):")
    print(f"  {'run':>6} | {'rank_전체':>8} | {'ens':>8} | {'z_gmm':>8} | {'z_mahal':>8} | {'z_spe':>8}")
    print(f"  {'-'*60}")

    zg_ser = pd.Series(zg_te, index=te_run_order)
    zm_ser = pd.Series(zm_te, index=te_run_order)
    zs_ser = pd.Series(zs_te, index=te_run_order)

    for run_id, score in top3_additions.items():
        rank_global = list(ens_ser.index).index(run_id) + 1
        print(f"  run{run_id:3d} | {rank_global:8d} | {score:8.4f} | {zg_ser[run_id]:8.4f} | {zm_ser[run_id]:8.4f} | {zs_ser[run_id]:8.4f}")

    # ── 새 예측 생성: Exp30 + 상위 3개 ─────────────────────────
    new_anom_set = exp30_anom_set | set(top3_additions.index.tolist())
    print(f"\n최종 이상 run 수: {len(new_anom_set)} (= {len(exp30_anom_set)} + {len(top3_additions)})")

    new_pred_run = pd.Series(
        [1 if r in new_anom_set else 0 for r in te_run_order],
        index=te_run_order
    )
    pred_row = test_run_ids.map(new_pred_run).values

    out_path = OUTPUT_DIR / "output_exp34b(GMM-Mahal-SPE30-PP240-safe).csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"\n저장 완료: {out_path}")
    print("  Exp30 237개 보존 + 상위 3개 추가 = 240개")


if __name__ == "__main__":
    main()
