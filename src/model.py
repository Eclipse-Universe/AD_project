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
    """KMeans 클러스터 중심까지 유클리드 거리를 이상 점수로 사용.

    정상 데이터로만 학습 → 이상 데이터는 클러스터 중심에서 멀리 떨어진다는 가정.
    규칙: decision_function() → 낮을수록 이상 (거리를 음수화해 IF/LOF와 동일 방향으로 통일).

    한계: 유클리드 거리는 피처를 독립적으로 취급함 → 피처 간 상관이 깨지는 이상을 포착 못함.
    개선 버전: KMeansMahalanobisDetector 참고.
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


class KMeansMahalanobisDetector:
    """KMeans + 마할라노비스 거리 기반 이상 탐지.

    유클리드 거리(KMeansAnomalyDetector)의 두 가지 한계를 동시에 해결한다:

    1. 피처 스케일 문제: 분산이 큰 피처(xmv_5 std_ratio 18.46)가 거리를 지배하는 것을
       공분산 역행렬 Σ⁻¹이 정규화해 모든 피처가 균등하게 기여하도록 만든다.

    2. 상관 구조 미반영: 유클리드 거리는 xmv_7↔xmeas_12(r=1.0)처럼 강하게 상관된 피처 쌍의
       관계가 깨지는 이상을 포착하지 못한다. 마할라노비스 거리는 Σ⁻¹에 공분산 구조가 인코딩되어
       있어 "두 피처의 관계가 틀어진" 이상 run에 더 큰 거리 값을 부여한다.

    거리 공식: d(x, μ_k) = √[(x−μ_k)ᵀ Σ⁻¹ (x−μ_k)]
    Σ: 전체 훈련 데이터의 글로벌 공분산 행렬 (250,000행으로 안정적으로 추정 가능)
    이론적 근거: EDA에서 |왜도|<0.2 확인 → 다변량 정규분포 가정 성립 → 마할라노비스 거리가 최적

    규칙: decision_function() → 낮을수록 이상 (거리 음수화, IF/LOF와 동일 방향).
    """

    def __init__(self, n_clusters: int = 50, random_state: int = 42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.cov_inv: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "KMeansMahalanobisDetector":
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        self.model.fit(X_arr)
        # 글로벌 공분산 역행렬 추정 (훈련 데이터 전체 사용 → 안정적)
        cov = np.cov(X_arr, rowvar=False)          # (d, d)
        self.cov_inv = np.linalg.pinv(cov)         # 유사역행렬: 수치 불안정 방지
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        centers = self.model.cluster_centers_       # (k, d)
        # 각 군집 중심까지 마할라노비스 거리 계산 후 최솟값 선택
        # diff @ cov_inv: (n, d) @ (d, d) → (n, d)
        # element-wise * diff 후 합산 → (n,): 각 점의 마할라노비스 거리 제곱
        all_dists = np.empty((X_arr.shape[0], len(centers)))
        for j, c in enumerate(centers):
            diff = X_arr - c                        # (n, d) 브로드캐스트
            mahal_sq = (diff @ self.cov_inv * diff).sum(axis=1)
            all_dists[:, j] = np.sqrt(np.maximum(mahal_sq, 0))
        min_dists = all_dists.min(axis=1)
        return -min_dists                           # 음수화: 낮을수록 이상


class LocalMahalanobisDetector:
    """KMeans + 클러스터별 마할라노비스 거리 (Local Mahal).

    Global KMeansMahalanobisDetector의 한계:
      전체 훈련 데이터의 단일 Σ⁻¹ 사용 → 서로 다른 운전 조건의 공분산이 평균됨.

    TEP 물리 배경:
      반응 부하·공급 조건·냉각 상태가 다른 여러 운전 조건이 공존하므로
      각 조건에서 센서 상관 구조가 다르다. 이를 하나의 Global Σ⁻¹로 근사하면
      운전 조건 간 전환점 근처의 이상을 제대로 포착하지 못할 수 있다.

    Local Mahal:
      클러스터 k 소속 샘플로 Σ_k를 개별 추정.
      d(x, k) = sqrt[(x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k)]
      min_k d(x, k) 를 이상 점수로 사용.

    규칙: decision_function() → 낮을수록 이상 (거리 음수화, 다른 모델과 방향 통일).
    """

    def __init__(self, n_clusters: int = 50, reg: float = 1e-6,
                 random_state: int = 42):
        self.n_clusters  = n_clusters
        self.reg         = reg
        self.model       = KMeans(n_clusters=n_clusters,
                                  random_state=random_state, n_init="auto")
        self.cov_invs_: list[np.ndarray] = []
        self.centers_:  np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "LocalMahalanobisDetector":
        X_arr  = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        labels = self.model.fit_predict(X_arr)
        self.centers_ = self.model.cluster_centers_
        d = X_arr.shape[1]
        self.cov_invs_ = []
        for k in range(self.n_clusters):
            X_k = X_arr[labels == k]
            if len(X_k) > d + 1:
                cov_k = np.cov(X_k, rowvar=False) + self.reg * np.eye(d)
            else:
                # 샘플 부족 시 단위 행렬 (유클리드 거리로 대체)
                cov_k = np.eye(d)
            self.cov_invs_.append(np.linalg.pinv(cov_k))
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        all_dists = np.empty((X_arr.shape[0], self.n_clusters))
        for j, (c, cov_inv) in enumerate(zip(self.centers_, self.cov_invs_)):
            diff        = X_arr - c
            mahal_sq    = (diff @ cov_inv * diff).sum(axis=1)
            all_dists[:, j] = np.sqrt(np.maximum(mahal_sq, 0))
        return -all_dists.min(axis=1)  # 음수화: 낮을수록 이상


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
    weight_decay: float = 0.0,
) -> "TEPAutoencoder":
    """StandardScaler 정규화된 X를 받아 AE를 학습한다.

    X는 이미 scale_features()를 거친 값이어야 한다 — 스케일이 맞지 않으면
    큰 feature가 MSE를 지배해 재구성 오차가 왜곡된다.

    weight_decay: L2 정규화 계수. 샘플 수 대비 파라미터가 많을 때 과적합 방지용.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
