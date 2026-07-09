"""
stability_analysis.py — Exp30 앙상블 안정성 분석

분석 항목:
  1. 점수 마진: 237번째 vs 238번째 run의 앙상블 점수 차이 (경계 안정성)
  2. 부트스트랩 안정성: train 재샘플링 × 100회 → 상위 237 run 일치율
     - Core runs: 100회 중 95회+ 이상 판정 → 매우 안정적
     - Marginal runs: 50~94회 → 경계선
     - Unstable runs: <50회 → 불안정

  리스크 저감 전략:
  - 부트스트랩에서 불안정한 run이 현재 FP/FN에 집중되어 있다면
    → 그 run들이 과적합의 주 원인 후보

실행: cd /root/AD_project/src && python stability_analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from data_loader import load_train, load_test
from model import KMeansMahalanobisDetector
from preprocess import fit_scaler, scale_features, select_features

DATA_PATH         = Path("/root/AD_project/data")
OUTPUT_DIR        = Path("/root/AD_project/outputs")
RUN_CONTAMINATION = 0.32
RANDOM_SEED       = 42
N_BOOTSTRAP       = 100
TEST_TS           = 960


def sep_index(tr, te):
    return float((te.mean() - tr.mean()) / (tr.std() + 1e-10))


def ensemble_score(tr_rvec, te_rvec, tr_X, te_X,
                   train_run_ids, test_run_ids, seed=42):
    """GMM+Mahal+SPE 앙상블 스코어 계산 (train/test 분리)."""
    def z(tr_arr, te_arr):
        mu, sg = tr_arr.mean(), tr_arr.std() + 1e-10
        return (te_arr - mu) / sg, (tr_arr - mu) / sg

    # GMM
    gmm = GaussianMixture(n_components=2, covariance_type="tied",
                          reg_covar=1e-4, n_init=10, max_iter=300,
                          random_state=seed)
    gmm.fit(tr_rvec)
    zg_te, zg_tr = z(-gmm.score_samples(tr_rvec), -gmm.score_samples(te_rvec))

    # Mahal
    mah = KMeansMahalanobisDetector(n_clusters=50, random_state=seed)
    mah.fit(tr_rvec)
    zm_te, zm_tr = z(-mah.decision_function(tr_rvec), -mah.decision_function(te_rvec))

    # SPE k=30
    pca = PCA(n_components=30, random_state=seed)
    pca.fit(tr_X)
    def spe(X):
        return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)

    tr_run_ids_vals = train_run_ids if isinstance(train_run_ids, np.ndarray) else train_run_ids.values
    te_run_ids_vals = test_run_ids if isinstance(test_run_ids, np.ndarray) else test_run_ids.values

    tr_run_order = sorted(set(tr_run_ids_vals))
    te_run_order = sorted(set(te_run_ids_vals))

    tr_spe_s = pd.Series(spe(tr_X), index=tr_run_ids_vals).groupby(level=0).mean()
    te_spe_s = pd.Series(spe(te_X), index=te_run_ids_vals).groupby(level=0).mean()

    spe_tr = tr_spe_s.reindex(tr_run_order).values
    spe_te = te_spe_s.reindex(te_run_order).values
    zs_te, zs_tr = z(spe_tr, spe_te)

    ens_te = 0.6 * zg_te + 0.3 * zm_te + 0.1 * zs_te
    return ens_te, te_run_order


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

    tr_run_vecs = tr_X_df.groupby(train_run_ids.values).mean()
    te_run_vecs = te_X_df.groupby(test_run_ids.values).mean()
    tr_run_order = tr_run_vecs.index.tolist()
    te_run_order = te_run_vecs.index.tolist()

    # ── 1. Exp30 기준 점수 마진 분석 ──────────────────────────────
    print("\n[1] Exp30 점수 마진 분석...")
    ens_te, _ = ensemble_score(
        tr_run_vecs.values, te_run_vecs.values,
        tr_X, te_X, train_run_ids, test_run_ids, seed=RANDOM_SEED
    )
    ens_series = pd.Series(ens_te, index=te_run_order).sort_values(ascending=False)

    n_flag = int(np.round(RUN_CONTAMINATION * 740))  # 237
    thr_score = ens_series.iloc[n_flag - 1]   # 237번째 (이상 판정 경계)
    next_score = ens_series.iloc[n_flag]       # 238번째 (정상 판정 경계)
    margin = thr_score - next_score

    print(f"  판정 기준 run 수: {n_flag}")
    print(f"  237번째 run (이상 마지노선) 점수: {thr_score:.6f}")
    print(f"  238번째 run (정상 최고점)   점수: {next_score:.6f}")
    print(f"  마진: {margin:.6f}  (클수록 경계가 뚜렷하고 안정적)")

    print("\n  경계 근처 ±10 run 점수:")
    print("  순위  | run ID | 앙상블 점수 | 판정")
    for rank in range(max(0, n_flag - 10), min(len(ens_series), n_flag + 11)):
        run_id  = ens_series.index[rank]
        score   = ens_series.iloc[rank]
        label   = "이상 ✓" if rank < n_flag else "정상 ○"
        marker  = " ← 경계" if rank == n_flag - 1 or rank == n_flag else ""
        print(f"  {rank+1:5d} | run{run_id:4d} | {score:11.6f} | {label}{marker}")

    # ── 2. 부트스트랩 안정성 분석 ─────────────────────────────────
    print(f"\n[2] 부트스트랩 안정성 분석 (n={N_BOOTSTRAP}회, seed 변경)...")
    print("    train run을 복원 추출하여 모델 재학습 후 상위 237 run 일치율 계산")

    all_train_runs = np.array(tr_run_order)
    n_train_runs   = len(all_train_runs)  # 500

    # 카운터: 각 test run이 '이상'으로 판정된 횟수
    anomaly_count = np.zeros(740, dtype=int)
    test_run_arr  = np.array(te_run_order)  # [run ID 순서]

    for b in range(N_BOOTSTRAP):
        seed_b = RANDOM_SEED + b * 7
        rng    = np.random.default_rng(seed_b)

        # train run 복원 추출
        sampled_runs = rng.choice(all_train_runs, size=n_train_runs, replace=True)
        mask = np.isin(train_run_ids.values, sampled_runs)

        # 복원 추출된 run들의 raw index로 데이터 추출
        # (같은 run이 여러 번 뽑힌 경우 중복 포함)
        boot_rows = []
        boot_run_ids = []
        for i, rid in enumerate(sampled_runs):
            run_mask = (train_run_ids.values == rid)
            rows_for_run = tr_X_df.values[run_mask]
            # 복원 추출 시 중복된 run은 새 ID 부여 (모델이 같은 run 두 번 학습 허용)
            boot_rows.append(rows_for_run)
            boot_run_ids.extend([i * 10000 + rid] * len(rows_for_run))

        boot_tr_X   = np.vstack(boot_rows)
        boot_run_ids_arr = np.array(boot_run_ids)
        boot_tr_rvec = pd.DataFrame(boot_tr_X).groupby(boot_run_ids_arr).mean().values

        try:
            ens_b, _ = ensemble_score(
                boot_tr_rvec, te_run_vecs.values,
                boot_tr_X, te_X,
                boot_run_ids_arr, test_run_ids, seed=seed_b
            )
        except Exception:
            continue  # 수치적으로 불안정한 경우 스킵

        # 상위 237 판정
        thr_b   = np.quantile(ens_b, 1 - RUN_CONTAMINATION)
        pred_b  = (ens_b >= thr_b).astype(int)
        anomaly_count += pred_b

        if (b + 1) % 20 == 0:
            print(f"    {b+1}/{N_BOOTSTRAP} 완료...")

    # ── 결과 집계 ──────────────────────────────────────────────────
    stability = pd.Series(anomaly_count, index=test_run_arr)
    core_runs     = stability[stability >= 95].index.tolist()
    marginal_runs = stability[(stability >= 50) & (stability < 95)].index.tolist()
    unstable_runs = stability[stability < 50].index.tolist()

    # Exp30 판정
    exp30 = pd.read_csv(OUTPUT_DIR / "output_exp30(GMM-Mahal-SPE30).csv")
    exp30_pred = exp30["faultNumber"].values.reshape(-1, TEST_TS)[:, 0]
    exp30_anom = set(te_run_order[i] for i, v in enumerate(exp30_pred) if v == 1)
    exp30_norm = set(te_run_order[i] for i, v in enumerate(exp30_pred) if v == 0)

    print(f"\n  부트스트랩 결과 요약 (100회 중 이상 판정 횟수):")
    print(f"  Core runs     (≥95회): {len(core_runs):3d} runs")
    print(f"  Marginal runs (50~94): {len(marginal_runs):3d} runs")
    print(f"  Unstable runs (< 50): {len(unstable_runs):3d} runs")

    # 불안정 run vs Exp30 판정 교차
    unstable_in_anom = [r for r in unstable_runs if r in exp30_anom]
    unstable_in_norm = [r for r in unstable_runs if r in exp30_norm]
    marginal_in_anom = [r for r in marginal_runs if r in exp30_anom]
    marginal_in_norm = [r for r in marginal_runs if r in exp30_norm]

    print(f"\n  Exp30 이상 판정(237개) 중 불안정 run: {len(unstable_in_anom)}개 (= 잠재적 FP 후보)")
    print(f"  Exp30 정상 판정(503개) 중 불안정 run: {len(unstable_in_norm)}개 (= 잠재적 FN 후보)")
    print(f"  Exp30 이상 판정 중 마진 run: {len(marginal_in_anom)}개")
    print(f"  Exp30 정상 판정 중 마진 run: {len(marginal_in_norm)}개")

    if unstable_in_anom:
        print(f"\n  Exp30 이상 판정 중 불안정 run 목록 (이상 판정 횟수 / 100):")
        for r in sorted(unstable_in_anom, key=lambda x: stability[x]):
            print(f"    run{r:4d}: {stability[r]:3d}/100회 이상 판정")

    if unstable_in_norm:
        print(f"\n  Exp30 정상 판정 중 불안정 run 목록 (이상 판정 횟수 / 100):")
        for r in sorted(unstable_in_norm, key=lambda x: stability[x], reverse=True):
            print(f"    run{r:4d}: {stability[r]:3d}/100회 이상 판정")

    # ── 결론 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결론 요약")
    print("=" * 60)
    print(f"  점수 마진: {margin:.6f}")
    if margin > 0.1:
        print("  → 경계가 뚜렷함. 일반화 리스크 낮음")
    elif margin > 0.01:
        print("  → 경계가 어느 정도 뚜렷함. 주의 필요")
    else:
        print("  → 경계가 좁음. 일반화 리스크 있음")

    if len(unstable_runs) < 10:
        print(f"  부트스트랩: 불안정 run {len(unstable_runs)}개 — 모델이 안정적")
    else:
        print(f"  부트스트랩: 불안정 run {len(unstable_runs)}개 — 과적합 경계 run 존재")


if __name__ == "__main__":
    main()
