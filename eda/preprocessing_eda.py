"""
전처리 설계를 위한 EDA.

목적:
  1. 피처별 분포 형태(왜도/첨도) → Scaler 종류 결정
  2. Train vs Test 분포 이동(KS-test) → 핵심 이상 신호 피처 파악
  3. 상관관계 재검토(|r| >= 0.90) → 제거 후보 피처 파악
  4. 결과 요약 → docs/EDA_SUMMARY.md로 저장

실행: cd eda && python preprocessing_eda.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats

DATA_PATH = Path("/root/AD_project/data")
OUT_PATH = Path("/root/AD_project/eda")
DOCS_PATH = Path("/root/AD_project/docs")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")

NON_FEAT = ["faultNumber", "simulationRun", "sample"]
FEAT_COLS = [c for c in train.columns if c not in NON_FEAT]

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"분석 피처 수: {len(FEAT_COLS)}\n")

# ── 1. 피처별 분포 통계 (왜도·첨도 → Scaler 결정) ──────────────────────────
print("=" * 60)
print("1. 피처별 분포 통계 (Train 기준)")
print("=" * 60)

rows = []
for feat in FEAT_COLS:
    tr = train[feat]
    skew = float(tr.skew())
    kurt = float(tr.kurtosis())
    q1, q3 = tr.quantile(0.25), tr.quantile(0.75)
    iqr = q3 - q1
    outlier_rate = float(((tr < q1 - 1.5 * iqr) | (tr > q3 + 1.5 * iqr)).mean())
    rows.append({
        "feature": feat,
        "mean": tr.mean(),
        "std": tr.std(),
        "skewness": skew,
        "kurtosis": kurt,
        "outlier_rate": outlier_rate,
        # |왜도| > 1 이면 RobustScaler 권장, <= 1 이면 StandardScaler 무방
        "scaler_rec": "Robust" if abs(skew) > 1 else "Standard",
    })

dist_df = pd.DataFrame(rows).set_index("feature")
dist_df.to_csv(OUT_PATH / "feature_distribution_stats.csv")

robust_count = (dist_df["scaler_rec"] == "Robust").sum()
standard_count = (dist_df["scaler_rec"] == "Standard").sum()
print(f"Scaler 권장 집계: Standard={standard_count}개, Robust={robust_count}개")
print("\n[왜도 상위 10 — 분포가 가장 비대칭인 피처]")
print(dist_df["skewness"].abs().sort_values(ascending=False).head(10).round(3).to_string())
print("\n[이상치 비율 상위 10 (IQR 기준)]")
print(dist_df["outlier_rate"].sort_values(ascending=False).head(10).round(4).to_string())


# ── 2. Train vs Test 분포 이동 (KS-test) ────────────────────────────────────
print("\n" + "=" * 60)
print("2. Train vs Test 분포 이동 (Kolmogorov-Smirnov test)")
print("=" * 60)
print("해석: statistic 클수록 분포 차이 큼. p<0.05 이면 유의미한 이동.")

ks_rows = []
for feat in FEAT_COLS:
    stat, pval = stats.ks_2samp(train[feat].values, test[feat].values)
    std_ratio = test[feat].std() / (train[feat].std() + 1e-9)
    mean_shift = abs(test[feat].mean() - train[feat].mean()) / (train[feat].std() + 1e-9)
    ks_rows.append({
        "feature": feat,
        "ks_stat": stat,
        "ks_pval": pval,
        "std_ratio_test_train": std_ratio,
        "mean_shift_normalized": mean_shift,
        "significant": pval < 0.05,
    })

ks_df = pd.DataFrame(ks_rows).set_index("feature")
ks_df.to_csv(OUT_PATH / "feature_ks_test.csv")

sig_count = ks_df["significant"].sum()
print(f"\n유의미한 분포 이동(p<0.05): {sig_count}/{len(FEAT_COLS)}개 피처")
print("\n[KS statistic 상위 15 — train/test 분포 차이가 가장 큰 피처]")
top_ks = ks_df.sort_values("ks_stat", ascending=False).head(15)
print(top_ks[["ks_stat", "ks_pval", "std_ratio_test_train", "mean_shift_normalized"]].round(4).to_string())


# ── 3. 상관관계 재검토 (|r| >= 0.90) ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 고상관 피처 쌍 (|r| >= 0.90) — 제거 후보")
print("=" * 60)

corr_mat = train[FEAT_COLS].corr()
upper_mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
upper = corr_mat.abs().where(upper_mask)
high_corr = upper.stack()[upper.stack() >= 0.90].sort_values(ascending=False)

print(f"총 {len(high_corr)}쌍\n")
for (f1, f2), r_val in high_corr.items():
    direction = "+" if corr_mat.loc[f1, f2] > 0 else "-"
    print(f"  {f1} ↔ {f2}: r={direction}{abs(r_val):.4f}")

# 제거 권장 피처 (각 쌍에서 두 번째 피처를 후보로 지정)
remove_candidates = list(dict.fromkeys([f2 for (f1, f2), _ in high_corr.items()]))
print(f"\n제거 후보 피처 ({len(remove_candidates)}개): {remove_candidates}")


# ── 4. Scaler 비교 시각화 (왜도 상위 6개 피처) ──────────────────────────────
top_skew_feats = dist_df["skewness"].abs().sort_values(ascending=False).head(6).index.tolist()

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"{f}<br>skew={dist_df.loc[f,'skewness']:.2f}" for f in top_skew_feats],
)
positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

for (r, c), feat in zip(positions, top_skew_feats):
    tr_vals = train[feat].values
    fig.add_trace(go.Histogram(x=tr_vals, name="Train", opacity=0.7,
                               marker_color="steelblue", nbinsx=60,
                               showlegend=(r==1 and c==1)), row=r, col=c)
    te_vals = test[feat].values
    fig.add_trace(go.Histogram(x=te_vals, name="Test", opacity=0.5,
                               marker_color="tomato", nbinsx=60,
                               showlegend=(r==1 and c==1)), row=r, col=c)

fig.update_layout(
    title="분포 비교 — 왜도 상위 6개 피처 (Train vs Test)",
    barmode="overlay", height=600, width=1100,
)
fig.write_html(str(OUT_PATH / "skew_dist_comparison.html"))
print(f"\n저장: skew_dist_comparison.html")


# ── 5. KS-stat 바 차트 (전체 52개 피처) ─────────────────────────────────────
ks_sorted = ks_df.sort_values("ks_stat", ascending=True)
bar_colors = ["tomato" if sig else "steelblue" for sig in ks_sorted["significant"]]

fig2 = go.Figure(go.Bar(
    y=ks_sorted.index,
    x=ks_sorted["ks_stat"],
    orientation="h",
    marker_color=bar_colors,
))
fig2.update_layout(
    title="피처별 Train/Test 분포 이동 (KS statistic) — 빨강: p<0.05",
    xaxis_title="KS statistic (클수록 이동 큼)",
    height=900, width=800,
)
fig2.write_html(str(OUT_PATH / "ks_test_barplot.html"))
print("저장: ks_test_barplot.html")


# ── 6. 요약 문서 생성 ────────────────────────────────────────────────────────
summary_lines = [
    "# EDA 요약 — 전처리 설계 근거",
    "",
    f"분석일: 2026-07-02 | 피처 수: {len(FEAT_COLS)}",
    "",
    "## 1. Scaler 선택",
    "",
    f"- |왜도| > 1인 피처: **{robust_count}개** → RobustScaler 권장",
    f"- |왜도| ≤ 1인 피처: **{standard_count}개** → StandardScaler 무방",
    "- 과반이 정규분포에 가까우면 StandardScaler, 왜도 큰 피처가 많으면 RobustScaler.",
    "- **최종 선택은 아래 왜도 분포 결과 보고 결정할 것.**",
    "",
    "## 2. Train/Test 분포 이동 (KS-test)",
    "",
    f"- 유의미한 이동(p<0.05): {sig_count}/{len(FEAT_COLS)}개 피처",
    "- KS stat 상위 피처 = 이상 run 탐지에 핵심 신호일 가능성 높음.",
    "- 상위 15개:",
]
for feat in top_ks.index:
    row = top_ks.loc[feat]
    summary_lines.append(
        f"  - {feat}: KS={row['ks_stat']:.4f}, std_ratio={row['std_ratio_test_train']:.2f}x"
    )

summary_lines += [
    "",
    "## 3. 제거 후보 피처 (|r| ≥ 0.90)",
    "",
    f"- 고상관 쌍 수: {len(high_corr)}",
    f"- 제거 후보: {remove_candidates}",
    "- **완전 상관(r=1.000)인 쌍은 하나를 반드시 제거해야 한다.**",
    "  xmeas_12↔xmv_7, xmeas_15↔xmv_8 확인 필수.",
    "",
    "## 다음 단계",
    "",
    "1. 위 결과 보고 StandardScaler vs RobustScaler 결정",
    "2. 고상관 피처 제거 여부 결정",
    "3. 같은 전처리로 IF, LOF, KMeans, KNN 순으로 실험",
]

summary_text = "\n".join(summary_lines)
(DOCS_PATH / "EDA_SUMMARY.md").write_text(summary_text, encoding="utf-8")
print("\n저장: docs/EDA_SUMMARY.md")
print("\n=== preprocessing EDA 완료 ===")
