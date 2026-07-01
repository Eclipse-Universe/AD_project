"""
t-SNE로 train/test 분포 차이를 시각화한다.

실행: cd eda && python tsne_visualization.py
결과: eda/tsne.html (브라우저에서 열어서 확인)
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
DATA_PATH = Path("/root/AD_project/data")
OUTPUT_PATH = Path("/root/AD_project/eda/tsne.html")
N_SAMPLE_RUNS = 10  # train/test 각각에서 뽑을 simulationRun 수

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 데이터 로드
train_data = pd.read_csv(DATA_PATH / "train.csv")
test_data = pd.read_csv(DATA_PATH / "test.csv")

non_numeric_cols = ["faultNumber", "simulationRun", "sample"]
use_cols = train_data.columns.difference(non_numeric_cols)

# simulationRun 단위로 샘플링 (t-SNE는 전체 데이터에 적용하면 너무 오래 걸림)
train_sample_runs = np.random.choice(train_data["simulationRun"].unique(), size=N_SAMPLE_RUNS, replace=False)
test_sample_runs = np.random.choice(test_data["simulationRun"].unique(), size=N_SAMPLE_RUNS, replace=False)

train_data = train_data[train_data["simulationRun"].isin(train_sample_runs)]
test_data = test_data[test_data["simulationRun"].isin(test_sample_runs)]

train_numeric = train_data[use_cols]
test_numeric = test_data[use_cols]
train_run_labels = train_data["simulationRun"]
test_run_labels = test_data["simulationRun"]

# StandardScaler: train으로 fit, test는 transform만
scaler = StandardScaler()
scaled_train = pd.DataFrame(
    scaler.fit_transform(train_numeric),
    index=train_numeric.index,
    columns=train_numeric.columns,
)
scaled_test = pd.DataFrame(
    scaler.transform(test_numeric),
    index=test_numeric.index,
    columns=test_numeric.columns,
)
scaled_all = pd.concat([scaled_train, scaled_test])

# t-SNE 차원 축소 (52차원 → 2차원)
print("t-SNE 학습 중... (수 분 소요될 수 있음)")
X_embedded = TSNE(n_components=2, random_state=RANDOM_SEED).fit_transform(scaled_all)
X_embedded_df = pd.DataFrame(X_embedded, index=scaled_all.index)

# 시각화
trace_train = go.Scatter(
    x=X_embedded_df.loc[train_run_labels.index, 0],
    y=X_embedded_df.loc[train_run_labels.index, 1],
    text=train_run_labels.values,
    name="Train (정상)",
    mode="markers",
    marker=dict(color="steelblue", opacity=0.6),
)
trace_test = go.Scatter(
    x=X_embedded_df.loc[test_run_labels.index, 0],
    y=X_embedded_df.loc[test_run_labels.index, 1],
    text=test_run_labels.values,
    name="Test (정상/이상 혼합)",
    mode="markers",
    marker=dict(color="tomato", opacity=0.6),
)

fig = go.Figure(
    data=[trace_train, trace_test],
    layout=go.Layout(
        title="t-SNE: Train vs Test 분포 비교",
        hovermode="closest",
        width=1000,
        height=800,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    ),
)

fig.write_html(str(OUTPUT_PATH))
print(f"저장 완료 → {OUTPUT_PATH}")
print("브라우저에서 열어서 확인하세요.")
