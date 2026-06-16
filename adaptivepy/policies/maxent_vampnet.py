"""Maximum-entropy VAMPNet adaptive sampling policy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import entropy

from adaptivepy.models import Dataset, SeedResult
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats

DEFAULT_HIDDEN_LAYERS = [16, 32, 64, 128, 256, 128, 64, 32, 16]
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 2048
DEFAULT_EPOCHS = 100
DEFAULT_LAGTIME = 1
DEFAULT_DEVICE = "cpu"
DEFAULT_NUM_THREADS = 1
VAMPNET_EPSILON = 1e-12


def split_trajectories_from_dataset(dataset: Dataset) -> List[np.ndarray]:
    """Split a dataset feature matrix into per-trajectory arrays.

    Parameters
    ----------
    dataset : Dataset
        Loaded dataset with ``feature_matrix`` and ``traj_index_map``.

    Returns
    -------
    list of np.ndarray
        Feature arrays ordered by trajectory ID.
    """
    if dataset.feature_matrix is None:
        raise ValueError("Dataset feature_matrix is not initialized.")

    trajectories: List[np.ndarray] = []
    for traj_id in sorted(dataset.traj_index_map):
        start, end = dataset.traj_index_map[traj_id]
        trajectories.append(np.asarray(dataset.feature_matrix[start:end], dtype=float))
    return trajectories


def validate_trajectory_lengths(
    trajectories: Sequence[np.ndarray],
    lagtime: int,
) -> None:
    """Ensure each trajectory is long enough for lagged VAMPNet training.

    Parameters
    ----------
    trajectories : sequence of np.ndarray
        Per-trajectory feature arrays.
    lagtime : int
        Lag time in frames.

    Raises
    ------
    ValueError
        If any trajectory is too short for the requested lag time.
    """
    if lagtime < 1:
        raise ValueError("MaxEnt VAMPNet 'lagtime' must be >= 1.")

    for idx, traj in enumerate(trajectories):
        if traj.shape[0] <= lagtime:
            raise ValueError(
                f"Trajectory {idx} has {traj.shape[0]} frames, but lagtime "
                f"requires at least {lagtime + 1} frames."
            )


def compute_shannon_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute per-row Shannon entropy from softmax probabilities.

    Uses :func:`scipy.stats.entropy` with natural logarithm, matching the
    author implementation.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape ``(n_frames, n_states)``.

    Returns
    -------
    np.ndarray
        Entropy values with shape ``(n_frames,)``.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")
    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative.")
    return entropy(probs, axis=1)


def rank_frames_by_entropy(
    entropy_scores: np.ndarray,
    global_indices: Sequence[int],
    n_seeds: int,
) -> List[int]:
    """Rank frame row indices by descending entropy.

    Ties are broken by ascending ``global_index`` for deterministic ordering.

    Parameters
    ----------
    entropy_scores : np.ndarray
        Entropy value per candidate frame.
    global_indices : sequence of int
        Global frame index for each entropy score.
    n_seeds : int
        Number of frames to select.

    Returns
    -------
    list of int
        Selected row indices into ``entropy_scores``.
    """
    if len(entropy_scores) != len(global_indices):
        raise ValueError("entropy_scores and global_indices must have equal length.")

    n_select = min(int(n_seeds), len(entropy_scores))
    if n_select == 0:
        return []

    ranked = sorted(
        range(len(entropy_scores)),
        key=lambda idx: (-float(entropy_scores[idx]), int(global_indices[idx])),
    )
    return ranked[:n_select]


def build_default_hidden_layers(n_features: int) -> List[int]:
    """Return the default hidden-layer widths from the author implementation."""
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    return list(DEFAULT_HIDDEN_LAYERS)


def _resolve_device(device_name: str, num_threads: int) -> Any:
    """Create a torch device and configure CPU thread count."""
    import torch

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(int(num_threads))
    return device


def build_vampnet_estimator(
    n_features: int,
    output_states: int,
    hidden_layers: Sequence[int],
    learning_rate: float,
    device_name: str,
    num_threads: int,
) -> Any:
    """Construct an untrained deeptime VAMPNet with a softmax output lobe."""
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.torch import MLP

    device = _resolve_device(device_name, num_threads)
    units = [int(n_features), *[int(width) for width in hidden_layers], int(output_states)]
    lobe = MLP(
        units=units,
        nonlinearity=nn.ReLU,
        output_nonlinearity=lambda: nn.Softmax(dim=-1),
    ).to(device)
    return VAMPNet(
        lobe=lobe,
        learning_rate=float(learning_rate),
        device=device,
        epsilon=VAMPNET_EPSILON,
    )


def fit_vampnet_estimator(
    estimator: Any,
    trajectories: Sequence[np.ndarray],
    lagtime: int,
    batch_size: int,
    epochs: int,
) -> Any:
    """Fit a VAMPNet estimator on lagged trajectory feature data."""
    from deeptime.util.data import TrajectoryDataset
    from torch.utils.data import DataLoader

    validate_trajectory_lengths(trajectories, lagtime)
    data_float32 = [np.asarray(traj, dtype=np.float32) for traj in trajectories]
    lagged_data = TrajectoryDataset.from_trajectories(int(lagtime), data_float32)
    loader = DataLoader(lagged_data, batch_size=int(batch_size), shuffle=True)
    estimator.fit(loader, n_epochs=int(epochs))
    return estimator


def transform_features(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Transform feature vectors into softmax state probabilities."""
    transformed = estimator.transform(np.asarray(features, dtype=np.float32))
    return np.asarray(transformed, dtype=float)


@register_policy
class MaxEntVampNetPolicy(Policy):
    """Select frames by Shannon entropy of VAMPNet soft state assignments.

    Implements the entropy-only MaxEnt VAMPNet acquisition function from
    Kleiman & Shukla (2023). Features are passed directly to a VAMPNet trained
    on lagged trajectory pairs; frames with the highest entropy of softmax
    state probabilities are selected as seeds. This policy does not require
    clustering.

    Parameters
    ----------
    output_states : int or None
        Number of softmax output nodes. Defaults to the feature dimensionality.
    lagtime : int
        Lag time in frames for VAMPNet training.
    hidden_layers : sequence of int or None
        Hidden MLP layer widths. Defaults to the author repository pattern.
    learning_rate : float
        VAMPNet learning rate.
    batch_size : int
        Training batch size.
    epochs : int
        Training epochs per policy invocation.
    device : str
        PyTorch device name, typically ``cpu`` or ``cuda``.
    num_threads : int
        CPU threads used by PyTorch during training.
    estimator : object or None
        Optional pre-fitted estimator for testing. When provided, training is
        skipped and this estimator is used for scoring.
  """

    name = "maxent_vampnet"
    requires_clustering = False

    def __init__(
        self,
        n_features: int,
        output_states: Optional[int] = None,
        lagtime: int = DEFAULT_LAGTIME,
        hidden_layers: Optional[Sequence[int]] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        device: str = DEFAULT_DEVICE,
        num_threads: int = DEFAULT_NUM_THREADS,
        estimator: Optional[Any] = None,
    ) -> None:
        if n_features < 1:
            raise ValueError("MaxEnt VAMPNet requires at least one feature dimension.")

        self.n_features = int(n_features)
        self.output_states = (
            int(output_states) if output_states is not None else self.n_features
        )
        if self.output_states < 2:
            raise ValueError("MaxEnt VAMPNet 'output_states' must be >= 2.")

        self.lagtime = int(lagtime)
        self.hidden_layers = (
            [int(width) for width in hidden_layers]
            if hidden_layers is not None
            else build_default_hidden_layers(self.n_features)
        )
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("MaxEnt VAMPNet 'hidden_layers' must be positive integers.")

        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.device = str(device)
        self.num_threads = int(num_threads)
        self._estimator = estimator
        self.last_scores: Dict[int, Dict[str, Any]] = {}

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Not used by this frame-level policy."""
        raise NotImplementedError(
            "MaxEnt VAMPNet selects frames directly; use select_frames instead."
        )

    def _fit_or_use_estimator(self, trajectories: Sequence[np.ndarray]) -> Any:
        if self._estimator is not None:
            return self._estimator
        estimator = build_vampnet_estimator(
            n_features=self.n_features,
            output_states=self.output_states,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            device_name=self.device,
            num_threads=self.num_threads,
        )
        return fit_vampnet_estimator(
            estimator,
            trajectories=trajectories,
            lagtime=self.lagtime,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def select_frames(self, dataset: Dataset, n_seeds: int) -> List[SeedResult]:
        """Train or reuse VAMPNet, score frames by entropy, and select seeds."""
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
            return []

        validate_trajectory_lengths(trajectories, self.lagtime)
        estimator = self._fit_or_use_estimator(trajectories)
        probabilities = transform_features(estimator, dataset.feature_matrix)
        entropy_scores = compute_shannon_entropy(probabilities)

        global_indices = [
            int(frame.global_index) if frame.global_index is not None else idx
            for idx, frame in enumerate(dataset.frames)
        ]
        selected_rows = rank_frames_by_entropy(
            entropy_scores,
            global_indices=global_indices,
            n_seeds=n_seeds,
        )
        selected_global = {global_indices[row] for row in selected_rows}

        self.last_scores = {}
        for row_idx, frame in enumerate(dataset.frames):
            global_index = global_indices[row_idx]
            probs = probabilities[row_idx]
            self.last_scores[global_index] = {
                "traj_id": int(frame.traj_id),
                "frame_id": int(frame.frame_id),
                "entropy": float(entropy_scores[row_idx]),
                "probabilities": probs.tolist(),
                "selected": global_index in selected_global,
            }

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
