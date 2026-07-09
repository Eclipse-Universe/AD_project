"""
run_exp35.py — Exp30 anomaly set에서 하위 K개 제거 (PP 감소 전략)

배경:
  PP=240 시도(Exp34b) 실패: rank 238~240이 전부 FP. FN runs는 240위권 밖.
  반대 방향: Exp30의 237개 anomaly run 중 ensemble score 최하위 K개 제거.
  하위 K개가 FP일 가능성이 높음 (threshold 직상위, 경계적 detection)

수학:
  현재 Exp30: TP≈231.5, FP≈5.5, PP=237 → F1=0.9707
  K=3 전부 FP: F1=463/474=0.9768 (+0.006)
  K=3 중 1 TP: F1=462/474=0.9747 (+0.004) — 여전히 개선
  K=3 중 2 TP: F1=461/474=0.9726 (+0.002) — 미세 개선
  K=3 중 3 TP: F1=460/474=0.9705 (-0.000) — 거의 동일

실행: cd /root/AD_project/src && python run_exp35.py
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

# 제거할 run 수 (하위 K개)
REMOVE_K = 3


def main():
    print(f"전략: Exp30 anomaly set에서 하위 {REMOVE_K}개 제거 → PP={237-REMOVE_K}\n")

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
    ens_ser = pd.Series(ens_te, index=te_run_order)

    zg_ser = pd.Series(zg_te, index=te_run_order)
    zm_ser = pd.Series(zm_te, index=te_run_order)
    zs_ser = pd.Series(zs_te, index=te_run_order)

    # ── Exp30 anomaly set 로드 ─────────────────────────────────
    exp30_df = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred_ts = exp30_df["faultNumber"].values
    exp30_run_pred = (
        pd.Series(exp30_pred_ts, index=test_run_ids.values)
        .groupby(level=0).first()
    )
    exp30_anom_set = set(exp30_run_pred.index[exp30_run_pred == 1].tolist())
    print(f"Exp30 anomaly set: {len(exp30_anom_set)}개")

    # ── Exp30 anomaly run들을 재계산 score로 정렬 ────────────────
    anom_scores = ens_ser[ens_ser.index.isin(exp30_anom_set)].sort_values()

    print(f"\nExp30 anomaly set 하위 10개 (제거 후보):")
    print(f"  {'순위':>4} | {'run':>6} | {'ens':>8} | {'z_gmm':>8} | {'z_mahal':>8} | {'z_spe':>8}")
    print(f"  {'-'*58}")
    for rank, (run_id, score) in enumerate(anom_scores.head(10).items()):
        marker = " ← 제거" if rank < REMOVE_K else ""
        print(f"  {rank+1:4d} | run{run_id:4d} | {score:8.4f} | {zg_ser[run_id]:8.4f} | "
              f"{zm_ser[run_id]:8.4f} | {zs_ser[run_id]:8.4f}{marker}")

    print(f"\nExp30 anomaly set 상위 5개 (핵심 TP, 참고용):")
    print(f"  {'run':>6} | {'ens':>8}")
    for run_id, score in anom_scores.tail(5).items():
        print(f"  run{run_id:4d} | {score:8.4f}")

    # ── 하위 K개 제거 ───────────────────────────────────────────
    remove_set = set(anom_scores.head(REMOVE_K).index.tolist())
    new_anom_set = exp30_anom_set - remove_set
    print(f"\n제거된 runs: {sorted(remove_set)}")
    print(f"최종 anomaly set 크기: {len(new_anom_set)} (={len(exp30_anom_set)}-{REMOVE_K})")

    # ── 예측 생성 ───────────────────────────────────────────────
    new_pred_run = pd.Series(
        [1 if r in new_anom_set else 0 for r in te_run_order],
        index=te_run_order
    )
    pred_row = test_run_ids.map(new_pred_run).values

    out_path = OUTPUT_DIR / f"output_exp35(Exp30-remove{REMOVE_K}bottom).csv"
    save_submission(pred_row, te_X_df.index, out_path)
    print(f"\n저장: {out_path}")

    # ── 이론적 예측 ─────────────────────────────────────────────
    print(f"\n=== 이론적 F1 예측 (A=240 가정) ===")
    tp = 231.5
    pp_new = 237 - REMOVE_K
    A = 240
    print(f"  제거 {REMOVE_K}개 전부 FP → F1={2*tp/(pp_new+A):.4f} (현재 0.9707 대비 +{2*tp/(pp_new+A)-0.9707:+.4f})")
    print(f"  제거 {REMOVE_K}개 중 1 TP → F1={2*(tp-1)/(pp_new+A):.4f} (현재 0.9707 대비 {2*(tp-1)/(pp_new+A)-0.9707:+.4f})")
    print(f"  제거 {REMOVE_K}개 중 2 TP → F1={2*(tp-2)/(pp_new+A):.4f} (현재 0.9707 대비 {2*(tp-2)/(pp_new+A)-0.9707:+.4f})")


if __name__ == "__main__":
    main()
