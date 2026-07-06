"""TS-DAR adaptive sampling policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from adaptivepy.models import Dataset, SeedResult
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.policies.maxent_vampnet import (
    split_trajectories_from_dataset,
    validate_trajectory_lengths,
)
from adaptivepy.stats.cluster_stats import ClusterStats

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 2048
DEFAULT_EPOCHS = 100
DEFAULT_PRETRAIN = 10
DEFAULT_LAGTIME = 1
DEFAULT_BETA = 0.01
DEFAULT_GAMMA = 1.0
DEFAULT_SCALING_TEMPERATURE = 0.1
DEFAULT_PROTO_UPDATE_FACTOR = 0.5
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_DEVICE = "cpu"
DEFAULT_NUM_THREADS = 1
DEFAULT_TRAIN_SPLIT = 0.9
TSDAR_EPSILON = 1e-6


@dataclass
class TsDarScoreResult:
    """Per-frame TS-DAR scoring result."""

    ood_scores: np.ndarray
    states: np.ndarray
    embeddings: np.ndarray
    probabilities: np.ndarray
    state_centers: np.ndarray


def default_n_states(n_features: int) -> int:
    """Return the default number of TS-DAR metastable states."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    return min(max(2, int(n_features)), 4)


def default_latent_dim(n_states: int) -> int:
    """Return the paper default latent dimension for a state count."""
    return 2 if int(n_states) <= 3 else 3


def default_encoder_sizes(n_features: int, latent_dim: int) -> List[int]:
    """Return default TS-DAR encoder layer sizes."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive.")
    return [int(n_features), 128, 64, int(latent_dim)]


def create_lagged_arrays(
    trajectories: Sequence[np.ndarray],
    lagtime: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create concatenated time-instantaneous and time-lagged training arrays."""
    validate_trajectory_lengths(trajectories, lagtime)
    x0_rows = []
    x1_rows = []
    for traj in trajectories:
        array = np.asarray(traj, dtype=np.float32)
        x0_rows.append(array[:-lagtime])
        x1_rows.append(array[lagtime:])
    x0 = np.concatenate(x0_rows, axis=0)
    x1 = np.concatenate(x1_rows, axis=0)
    if x0.shape[0] < 2:
        raise ValueError("TS-DAR requires at least two lagged frame pairs.")
    return x0, x1


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "TS-DAR requires PyTorch. Install AdaptivePy with the 'torch' extra."
        ) from exc
    return torch


def _resolve_device(device_name: str, num_threads: int) -> Any:
    torch = _require_torch()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(int(num_threads))
    return device


def _set_torch_seed(random_state: Optional[int]) -> None:
    if random_state is None:
        return
    torch = _require_torch()
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_tsdar_lobe(
    encoder_sizes: Sequence[int],
    n_states: int,
    gamma: float = DEFAULT_GAMMA,
) -> Any:
    """Build the TS-DAR neural network lobe."""
    torch = _require_torch()
    import torch.nn as nn

    sizes = [int(size) for size in encoder_sizes]
    if len(sizes) < 2:
        raise ValueError("TS-DAR encoder_sizes must contain at least input and latent sizes.")
    if int(n_states) < 2:
        raise ValueError("TS-DAR n_states must be >= 2.")

    class _TsDarLobe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: List[Any] = [nn.BatchNorm1d(sizes[0])]
            for idx in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
                if idx < len(sizes) - 2:
                    layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*layers)
            self.classifier = nn.Linear(sizes[-1], int(n_states), bias=True)
            self.softmax = nn.Softmax(dim=-1)
            self.n_states = int(n_states)
            self.gamma = float(gamma)
            self.hypersphere_embs: Optional[Any] = None
            self.logits: Optional[Any] = None
            self.probs: Optional[Any] = None

        def forward(self, x: Any) -> Any:
            embeddings = self.encoder(x)
            norms = torch.linalg.norm(embeddings, dim=-1, keepdim=True)
            embeddings = self.gamma * embeddings / torch.clamp(norms, min=TSDAR_EPSILON)
            self.hypersphere_embs = embeddings
            self.logits = self.classifier(embeddings)
            self.probs = self.softmax(self.logits)
            return self.probs

    return _TsDarLobe()


def compute_covariance_matrix(x: Any, y: Any, remove_mean: bool = True) -> tuple[Any, Any, Any]:
    """Compute covariance matrices used by the VAMP-2 score."""
    if x.shape[0] < 2:
        raise ValueError("VAMP-2 covariance requires at least two samples.")
    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
    scale = 1.0 / float(x.shape[0] - 1)
    cov_00 = scale * x.t().matmul(x)
    cov_01 = scale * x.t().matmul(y)
    cov_11 = scale * y.t().matmul(y)
    return cov_00, cov_01, cov_11


def _matrix_inverse(matrix: Any, epsilon: float, return_sqrt: bool) -> Any:
    torch = _require_torch()
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    matrix = matrix + float(epsilon) * identity
    eigval, eigvec = torch.linalg.eigh(matrix)
    eigval = torch.clamp(torch.abs(eigval), min=float(epsilon))
    powers = torch.rsqrt(eigval) if return_sqrt else torch.reciprocal(eigval)
    return eigvec.matmul(torch.diag(powers)).matmul(eigvec.t())


def vamp2_loss(
    probabilities_0: Any,
    probabilities_1: Any,
    epsilon: float = TSDAR_EPSILON,
    symmetrized: bool = False,
) -> Any:
    """Return negative VAMP-2 score for optimization."""
    cov_00, cov_01, cov_11 = compute_covariance_matrix(
        probabilities_0, probabilities_1
    )
    if symmetrized:
        cov_0 = 0.5 * (cov_00 + cov_11)
        cov_1 = 0.5 * (cov_01 + cov_01.t())
        cov_0_inv_sqrt = _matrix_inverse(cov_0, epsilon, return_sqrt=True)
        koopman = cov_0_inv_sqrt.matmul(cov_1).matmul(cov_0_inv_sqrt).t()
    else:
        cov_00_inv_sqrt = _matrix_inverse(cov_00, epsilon, return_sqrt=True)
        cov_11_inv_sqrt = _matrix_inverse(cov_11, epsilon, return_sqrt=True)
        koopman = cov_00_inv_sqrt.matmul(cov_01).matmul(cov_11_inv_sqrt).t()
    score = torch_norm_fro(koopman) ** 2 + 1.0
    return -score


def torch_norm_fro(matrix: Any) -> Any:
    """Compute a Frobenius norm without importing torch at module import time."""
    torch = _require_torch()
    return torch.linalg.norm(matrix, ord="fro")


class _DispersionLoss:
    """State-center dispersion regularizer with EMA prototype updates."""

    def __init__(
        self,
        n_states: int,
        latent_dim: int,
        device: Any,
        proto_update_factor: float,
        scaling_temperature: float,
    ) -> None:
        torch = _require_torch()
        self.n_states = int(n_states)
        self.device = device
        self.proto_update_factor = float(proto_update_factor)
        self.scaling_temperature = float(scaling_temperature)
        self.prototypes = torch.zeros(self.n_states, int(latent_dim), device=device)

    def __call__(self, features: Any, labels: Any) -> Any:
        torch = _require_torch()
        import torch.nn.functional as F

        labels = labels.view(-1)
        one_hot = torch.zeros(features.shape[0], self.n_states, device=self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        counts = one_hot.sum(dim=0)
        mask = counts > 0
        counts_safe = torch.where(mask, counts, torch.ones_like(counts))
        means = one_hot.t().matmul(features) / counts_safe.unsqueeze(1)
        updated = self.proto_update_factor * self.prototypes + (
            1.0 - self.proto_update_factor
        ) * means
        prototypes = self.prototypes.clone()
        prototypes[mask] = F.normalize(updated[mask], dim=1)
        self.prototypes = prototypes.detach()

        logits = prototypes.matmul(prototypes.t()) / self.scaling_temperature
        negative_mask = (
            1.0
            - torch.eye(self.n_states, dtype=features.dtype, device=self.device)
        )
        counts_negative = negative_mask.sum(dim=1)
        loss = torch.log((negative_mask * torch.exp(logits)).sum(dim=1) / counts_negative)
        loss = loss[~torch.isnan(loss)]
        if loss.numel() == 0:
            return torch.zeros((), dtype=features.dtype, device=self.device)
        return loss.mean()


class _TsDarEstimator:
    """Fitted TS-DAR estimator used for frame scoring."""

    def __init__(
        self,
        lobe: Any,
        device: Any,
        batch_size: int,
    ) -> None:
        self.lobe = lobe
        self.device = device
        self.batch_size = int(batch_size)

    def _transform_array(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = _require_torch()
        self.lobe.eval()
        probabilities = []
        embeddings = []
        with torch.no_grad():
            array = np.asarray(features, dtype=np.float32)
            for start in range(0, array.shape[0], self.batch_size):
                batch = torch.from_numpy(array[start : start + self.batch_size]).to(
                    self.device
                )
                probs = self.lobe(batch)
                probabilities.append(probs.detach().cpu().numpy())
                embeddings.append(
                    self.lobe.hypersphere_embs.detach().cpu().numpy()
                )
        probs_array = np.concatenate(probabilities, axis=0)
        emb_array = np.concatenate(embeddings, axis=0)
        states = np.argmax(probs_array, axis=1)
        return probs_array, emb_array, states

    def score(self, trajectories: Sequence[np.ndarray]) -> TsDarScoreResult:
        features = np.concatenate([np.asarray(traj, dtype=np.float32) for traj in trajectories])
        probabilities, embeddings, states = self._transform_array(features)
        state_centers = compute_state_centers(embeddings, states, self.lobe.n_states)
        ood_scores = compute_ood_scores(embeddings, state_centers)
        return TsDarScoreResult(
            ood_scores=ood_scores,
            states=states,
            embeddings=embeddings,
            probabilities=probabilities,
            state_centers=state_centers,
        )


def compute_state_centers(
    embeddings: np.ndarray,
    states: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Compute normalized state-center vectors from hyperspherical embeddings."""
    emb = np.asarray(embeddings, dtype=float)
    labels = np.asarray(states, dtype=int)
    centers = np.zeros((int(n_states), emb.shape[1]), dtype=float)
    for state in range(int(n_states)):
        mask = labels == state
        if not np.any(mask):
            continue
        center = emb[mask].sum(axis=0)
        norm = np.linalg.norm(center)
        if norm > 0:
            centers[state] = center / norm
    return centers


def compute_ood_scores(
    embeddings: np.ndarray,
    state_centers: np.ndarray,
) -> np.ndarray:
    """Compute TS-DAR OOD scores from cosine distance to nearest state center."""
    emb = np.asarray(embeddings, dtype=float)
    centers = np.asarray(state_centers, dtype=float)
    emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
    center_norm = np.linalg.norm(centers, axis=1, keepdims=True).T
    normalized_emb = emb / np.clip(emb_norm, TSDAR_EPSILON, None)
    normalized_centers = centers / np.clip(center_norm.T, TSDAR_EPSILON, None)
    similarities = normalized_emb.dot(normalized_centers.T)
    return 1.0 - np.max(similarities, axis=1)


def rank_frames_by_ood(
    ood_scores: np.ndarray,
    global_indices: Sequence[int],
    n_seeds: int,
) -> List[int]:
    """Rank frame row indices by descending OOD score."""
    if len(ood_scores) != len(global_indices):
        raise ValueError("ood_scores and global_indices must have equal length.")
    n_select = min(int(n_seeds), len(ood_scores))
    if n_select == 0:
        return []
    ranked = sorted(
        range(len(ood_scores)),
        key=lambda idx: (-float(ood_scores[idx]), int(global_indices[idx])),
    )
    return ranked[:n_select]


def fit_tsdar_estimator(
    trajectories: Sequence[np.ndarray],
    n_features: int,
    n_states: int,
    latent_dim: int,
    encoder_sizes: Sequence[int],
    lagtime: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    pretrain: int,
    beta: float,
    gamma: float,
    scaling_temperature: float,
    proto_update_factor: float,
    optimizer: str,
    device_name: str,
    num_threads: int,
    train_split: float,
    random_state: Optional[int],
) -> _TsDarEstimator:
    """Train a TS-DAR estimator on lagged trajectory feature arrays."""
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset, random_split

    _set_torch_seed(random_state)
    device = _resolve_device(device_name, num_threads)
    x0, x1 = create_lagged_arrays(trajectories, lagtime)
    dataset = TensorDataset(torch.from_numpy(x0), torch.from_numpy(x1))

    if train_split < 1.0:
        n_train = max(2, int(round(len(dataset) * float(train_split))))
        n_train = min(n_train, len(dataset))
        n_val = len(dataset) - n_train
        if n_val > 0:
            generator = torch.Generator().manual_seed(int(random_state or 0))
            train_data, _ = random_split(dataset, [n_train, n_val], generator=generator)
        else:
            train_data = dataset
    else:
        train_data = dataset

    loader = DataLoader(
        train_data,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    lobe = build_tsdar_lobe(encoder_sizes, n_states=n_states, gamma=gamma).to(device)
    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
    }
    if optimizer not in optimizers:
        raise ValueError("TS-DAR optimizer must be one of: Adam, SGD, RMSprop.")
    optimizer_obj = optimizers[optimizer](lobe.parameters(), lr=float(learning_rate))
    dispersion = _DispersionLoss(
        n_states=n_states,
        latent_dim=latent_dim,
        device=device,
        proto_update_factor=proto_update_factor,
        scaling_temperature=scaling_temperature,
    )

    for epoch in range(int(epochs)):
        lobe.train()
        for batch_0, batch_1 in loader:
            if batch_0.shape[0] < 2:
                continue
            batch_0 = batch_0.to(device=device)
            batch_1 = batch_1.to(device=device)
            optimizer_obj.zero_grad()
            probs_0 = lobe(batch_0)
            embeddings_0 = lobe.hypersphere_embs
            probs_1 = lobe(batch_1)
            loss = vamp2_loss(probs_0, probs_1)
            if epoch >= int(pretrain):
                labels_0 = torch.argmax(probs_0, dim=-1).detach()
                loss = loss + float(beta) * dispersion(embeddings_0, labels_0)
            loss.backward()
            optimizer_obj.step()

    return _TsDarEstimator(lobe=lobe, device=device, batch_size=batch_size)


@register_policy
class TsDarPolicy(Policy):
    """Select frames with high TS-DAR out-of-distribution scores."""

    name = "ts_dar"
    requires_clustering = False

    def __init__(
        self,
        n_features: int,
        n_states: Optional[int] = None,
        latent_dim: Optional[int] = None,
        encoder_sizes: Optional[Sequence[int]] = None,
        lagtime: int = DEFAULT_LAGTIME,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        pretrain: int = DEFAULT_PRETRAIN,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        scaling_temperature: float = DEFAULT_SCALING_TEMPERATURE,
        proto_update_factor: float = DEFAULT_PROTO_UPDATE_FACTOR,
        optimizer: str = DEFAULT_OPTIMIZER,
        device: str = DEFAULT_DEVICE,
        num_threads: int = DEFAULT_NUM_THREADS,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        random_state: Optional[int] = None,
        estimator: Optional[Any] = None,
    ) -> None:
        if n_features < 1:
            raise ValueError("TS-DAR requires at least one feature dimension.")
        self.n_features = int(n_features)
        self.n_states = int(n_states) if n_states is not None else default_n_states(n_features)
        if self.n_states < 2:
            raise ValueError("TS-DAR 'n_states' must be >= 2.")
        self.latent_dim = (
            int(latent_dim) if latent_dim is not None else default_latent_dim(self.n_states)
        )
        if self.latent_dim < 1:
            raise ValueError("TS-DAR 'latent_dim' must be >= 1.")
        self.encoder_sizes = (
            [int(size) for size in encoder_sizes]
            if encoder_sizes is not None
            else default_encoder_sizes(self.n_features, self.latent_dim)
        )
        if self.encoder_sizes[0] != self.n_features:
            raise ValueError("TS-DAR encoder_sizes first value must equal n_features.")
        if self.encoder_sizes[-1] != self.latent_dim:
            raise ValueError("TS-DAR encoder_sizes last value must equal latent_dim.")
        self.lagtime = int(lagtime)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.pretrain = int(pretrain)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.scaling_temperature = float(scaling_temperature)
        self.proto_update_factor = float(proto_update_factor)
        self.optimizer = str(optimizer)
        self.device = str(device)
        self.num_threads = int(num_threads)
        self.train_split = float(train_split)
        self.random_state = random_state
        self._estimator = estimator
        self.last_scores: Dict[int, Dict[str, Any]] = {}
        self.last_state_centers: List[List[float]] = []

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Not used by this frame-level policy."""
        raise NotImplementedError("TS-DAR selects frames directly; use select_frames.")

    def _fit_or_use_estimator(self, trajectories: Sequence[np.ndarray]) -> Any:
        if self._estimator is not None:
            return self._estimator
        return fit_tsdar_estimator(
            trajectories=trajectories,
            n_features=self.n_features,
            n_states=self.n_states,
            latent_dim=self.latent_dim,
            encoder_sizes=self.encoder_sizes,
            lagtime=self.lagtime,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            pretrain=self.pretrain,
            beta=self.beta,
            gamma=self.gamma,
            scaling_temperature=self.scaling_temperature,
            proto_update_factor=self.proto_update_factor,
            optimizer=self.optimizer,
            device_name=self.device,
            num_threads=self.num_threads,
            train_split=self.train_split,
            random_state=self.random_state,
        )

    def select_frames(self, dataset: Dataset, n_seeds: int) -> List[SeedResult]:
        """Train or reuse TS-DAR, score frames by OOD score, and select seeds."""
        if dataset.feature_matrix is None:
            raise ValueError("Dataset feature_matrix is not initialized.")
        if dataset.feature_matrix.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got "
                f"{dataset.feature_matrix.shape[1]}."
            )

        trajectories = split_trajectories_from_dataset(dataset)
        if not trajectories:
            self.last_scores = {}
            self.last_state_centers = []
            return []
        validate_trajectory_lengths(trajectories, self.lagtime)
        estimator = self._fit_or_use_estimator(trajectories)
        score_result = estimator.score(trajectories)

        global_indices = [
            int(frame.global_index) if frame.global_index is not None else idx
            for idx, frame in enumerate(dataset.frames)
        ]
        selected_rows = rank_frames_by_ood(
            score_result.ood_scores,
            global_indices=global_indices,
            n_seeds=n_seeds,
        )
        selected_global = {global_indices[row] for row in selected_rows}

        self.last_scores = {}
        for row_idx, frame in enumerate(dataset.frames):
            global_index = global_indices[row_idx]
            self.last_scores[global_index] = {
                "traj_id": int(frame.traj_id),
                "frame_id": int(frame.frame_id),
                "ood_score": float(score_result.ood_scores[row_idx]),
                "state": int(score_result.states[row_idx]),
                "embedding": score_result.embeddings[row_idx].tolist(),
                "probabilities": score_result.probabilities[row_idx].tolist(),
                "selected": global_index in selected_global,
            }
        self.last_state_centers = score_result.state_centers.tolist()

        seeds: List[SeedResult] = []
        for seed_id, row_idx in enumerate(selected_rows):
            frame = dataset.frames[row_idx]
            global_index = global_indices[row_idx]
            seeds.append(
                SeedResult(
                    seed_id=seed_id,
                    policy=self.name,
                    traj_id=int(frame.traj_id),
                    frame_id=int(frame.frame_id),
                    cluster_id=frame.cluster_id,
                    global_index=global_index,
                )
            )
        return seeds
