"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 49와 동일한 역할(모델 생성+fit).

하이퍼파라미터는 함수 인자로 받는다 — Exp 1의 contamination 조정이 호출부 한 줄만
바뀌면 되게 하기 위함 (docs/EXPERIMENT_LOG.md Exp 0 다음 계획 참고).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset


def train_isolation_forest(X: pd.DataFrame, random_state: int, **params) -> IsolationForest:
    model = IsolationForest(random_state=random_state, **params)
    model.fit(X)
    return model


def train_sgd_ocsvm(X: pd.DataFrame, random_state: int, **params) -> SGDOneClassSVM:
    # 백로그 항목 (EXPERIMENT_LOG.md 참고) — run_experiment.py에서는 아직 호출 안 함
    model = SGDOneClassSVM(random_state=random_state, **params)
    model.fit(X)
    return model


def train_lof(X: pd.DataFrame, **params) -> LocalOutlierFactor:
    """LocalOutlierFactor — novelty=True 필수 (test에 decision_function 적용하려면).

    규칙: decision_function() → 낮을수록 이상. IF와 동일 방향.
    """
    model = LocalOutlierFactor(novelty=True, **params)
    model.fit(X)
    return model


class KMeansAnomalyDetector:
    """KMeans 클러스터 중심까지 거리를 이상 점수로 사용.

    정상 데이터로만 학습 → 이상 데이터는 클러스터 중심에서 멀리 떨어진다는 가정.
    규칙: decision_function() → 낮을수록 이상 (거리를 음수화해 IF/LOF와 동일 방향으로 통일).
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

    def fit(self, X: pd.DataFrame) -> "KMeansAnomalyDetector":
        self.model.fit(X)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        dists = self.model.transform(X.values if hasattr(X, "values") else X)
        min_dists = dists.min(axis=1)
        return -min_dists  # 음수화: 낮을수록(= 거리가 멀수록) 이상


class KNNAnomalyDetector:
    """k-NN 거리 기반 이상 탐지.

    정상 데이터 k개 이웃까지의 평균 거리를 이상 점수로 사용.
    규칙: decision_function() → 낮을수록 이상 (거리 음수화).
    """

    def __init__(self, n_neighbors: int = 5):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")

    def fit(self, X: pd.DataFrame) -> "KNNAnomalyDetector":
        self.model.fit(X)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        dists, _ = self.model.kneighbors(X.values if hasattr(X, "values") else X)
        mean_dists = dists.mean(axis=1)
        return -mean_dists  # 음수화: 낮을수록(= 멀수록) 이상


class TEPAutoencoder(nn.Module):
    """정상 데이터만으로 학습, 재구성 오차를 이상 점수로 사용.

    IF가 놓치는 피처 간 상관관계 패턴(예: xmeas_12↔xmv_7 완전 상관)을 bottleneck
    압축-복원 과정에서 암묵적으로 학습한다. 상관이 깨지는 이상 run은 복원 오차가 높게 나온다.

    입출력: StandardScaler 정규화된 float32 텐서 (배치 × input_dim)
    """

    def __init__(self, input_dim: int = 52, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]

        # Encoder: input_dim → ... → bottleneck
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: bottleneck → ... → input_dim (마지막 레이어는 활성화 없음)
        dec_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        dec_layers = []
        prev = hidden_dims[-1]
        for i, h in enumerate(dec_dims):
            dec_layers.append(nn.Linear(prev, h))
            if i < len(dec_dims) - 1:
                dec_layers.append(nn.ReLU())
            prev = h
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)


def train_autoencoder(
    X: pd.DataFrame,
    hidden_dims: list = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
    patience: int = 10,
    random_state: int = 42,
) -> "TEPAutoencoder":
    """StandardScaler 정규화된 X를 받아 AE를 학습한다.

    X는 이미 scale_features()를 거친 값이어야 한다 — 스케일이 맞지 않으면
    큰 feature가 MSE를 지배해 재구성 오차가 왜곡된다.
    """
    if hidden_dims is None:
        hidden_dims = [32, 16, 8]

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_np = X.values.astype(np.float32)
    n_val = int(len(X_np) * val_ratio)
    X_val_t = torch.tensor(X_np[:n_val])
    X_train_t = torch.tensor(X_np[n_val:])

    loader = DataLoader(
        TensorDataset(X_train_t), batch_size=batch_size, shuffle=True
    )

    model = TEPAutoencoder(input_dim=X.shape[1], hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), X_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop @ epoch {epoch}, best val loss: {best_val_loss:.6f}")
                break

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | val loss: {val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return model
