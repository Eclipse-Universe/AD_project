"""
모델링 전에 했어야 할 EDA.
t-SNE 이후 추가로 확인하는 분석.

실행: cd eda && python eda_analysis.py
결과: eda/correlation_heatmap.html, eda/timeseries_sample.html, eda/feature_dist.html
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px

DATA_PATH = Path("/root/AD_project/data")
OUT_PATH = Path("/root/AD_project/eda")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")

NON_FEAT = ["faultNumber", "simulationRun", "sample"]
FEAT_COLS = [c for c in train.columns if c not in NON_FEAT]

# ── 1. 상관관계 히트맵 ───────────────────────────────────────────────────────
corr = train[FEAT_COLS].corr()

fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale="RdBu",
    zmid=0,
    zmin=-1, zmax=1,
    colorbar=dict(title="Pearson r"),
))
fig_corr.update_layout(
    title="Feature Correlation Heatmap (Train)",
    width=900, height=850,
)
fig_corr.write_html(str(OUT_PATH / "correlation_heatmap.html"))
print("저장: correlation_heatmap.html")

# ── 2. 완전 상관(|r| >= 0.99) 쌍 출력 ──────────────────────────────────────
upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = upper.stack()[upper.stack() >= 0.99].sort_values(ascending=False)
print("\n[|r| >= 0.99 피처 쌍]")
print(high_corr.to_string())

# ── 3. 분산 하위 피처 (정상 상태에서 거의 안 변하는 것들) ──────────────────
var_sorted = train[FEAT_COLS].var().sort_values()
print("\n[분산 하위 10개 — 정상 상태에서 가장 안 변하는 피처]")
print(var_sorted.head(10).to_string())
print("\n[분산 상위 10개 — 가장 많이 변하는 피처]")
print(var_sorted.tail(10).to_string())

# ── 4. 분산 하위 피처가 test에서 어떻게 변하는지 비교 ──────────────────────
low_var_feats = var_sorted.head(5).index.tolist()
print("\n[분산 하위 5개 피처: train vs test 통계 비교]")
for feat in low_var_feats:
    t_std = train[feat].std()
    te_std = test[feat].std()
    print(f"  {feat}: train std={t_std:.4f}, test std={te_std:.4f}, 배율={te_std/t_std:.1f}x")

# ── 5. 시계열 시각화: 정상 run 1개 + test run 2개(외곽/중앙 각 1개) ────────
# 시각적으로 잘 보이는 피처를 고분산 기준으로 선택
top_feats = var_sorted.tail(6).index.tolist()

train_run = train[train["simulationRun"] == train["simulationRun"].unique()[0]]
test_runs = test["simulationRun"].unique()
test_run_a = test[test["simulationRun"] == test_runs[0]]   # 임의 test run A
test_run_b = test[test["simulationRun"] == test_runs[5]]   # 임의 test run B

fig_ts = make_subplots(rows=len(top_feats), cols=1,
                       shared_xaxes=True,
                       subplot_titles=top_feats)

for i, feat in enumerate(top_feats, start=1):
    fig_ts.add_trace(go.Scatter(y=train_run[feat].values, name="Train run (정상)",
                                line=dict(color="steelblue"),
                                showlegend=(i == 1)), row=i, col=1)
    fig_ts.add_trace(go.Scatter(y=test_run_a[feat].values, name="Test run A",
                                line=dict(color="tomato"),
                                showlegend=(i == 1)), row=i, col=1)
    fig_ts.add_trace(go.Scatter(y=test_run_b[feat].values, name="Test run B",
                                line=dict(color="orange"),
                                showlegend=(i == 1)), row=i, col=1)

fig_ts.update_layout(
    title="시계열 비교: Train(정상) vs Test run A/B — 고분산 피처",
    height=200 * len(top_feats),
    width=1000,
)
fig_ts.write_html(str(OUT_PATH / "timeseries_sample.html"))
print("\n저장: timeseries_sample.html")

# ── 6. Train vs Test 피처별 분포 비교 (분산 상위 6개) ──────────────────────
fig_dist = make_subplots(rows=2, cols=3,
                         subplot_titles=top_feats)
positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

for (r, c), feat in zip(positions, top_feats):
    fig_dist.add_trace(go.Histogram(x=train[feat], name="Train", opacity=0.6,
                                    marker_color="steelblue",
                                    showlegend=(r==1 and c==1),
                                    nbinsx=50), row=r, col=c)
    fig_dist.add_trace(go.Histogram(x=test[feat], name="Test", opacity=0.6,
                                    marker_color="tomato",
                                    showlegend=(r==1 and c==1),
                                    nbinsx=50), row=r, col=c)

fig_dist.update_layout(
    title="Train vs Test 분포 비교 (고분산 피처 6개)",
    barmode="overlay",
    height=600, width=1000,
)
fig_dist.write_html(str(OUT_PATH / "feature_dist.html"))
print("저장: feature_dist.html")
print("\n=== EDA 완료 ===")
