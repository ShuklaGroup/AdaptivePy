"""Core data models for AdaptivePy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FrameRecord:
    """A single frame tracked through the adaptive sampling pipeline.

    Attributes
    ----------
    traj_id : int
        Index of the source trajectory.
    frame_id : int
        Frame index within the source trajectory.
    features : np.ndarray
        Feature vector for this frame, shape ``(n_features,)``.
    cluster_id : int or None
        Assigned cluster label after clustering.
    global_index : int or None
        Row index in the concatenated feature matrix.
    """

    traj_id: int
    frame_id: int
    features: np.ndarray
    cluster_id: Optional[int] = None
    global_index: Optional[int] = None


@dataclass
class Dataset:
    """Internal representation of loaded trajectory features.

    Attributes
    ----------
    frames : list of FrameRecord
        One record per frame across all trajectories.
    feature_matrix : np.ndarray
        Concatenated features, shape ``(n_total_frames, n_features)``.
    traj_index_map : dict
        Maps ``traj_id`` to ``(start_index, end_index)`` in ``feature_matrix``.
    traj_names : list of str
        Basenames of feature files (without extension), e.g. ``traj_0``.
    """

    frames: List[FrameRecord] = field(default_factory=list)
    feature_matrix: Optional[np.ndarray] = None
    traj_index_map: Dict[int, tuple[int, int]] = field(default_factory=dict)
    traj_names: List[str] = field(default_factory=list)


@dataclass
class SeedResult:
    """A selected seed frame produced by a policy.

    Attributes
    ----------
    seed_id : int
        Sequential identifier within a policy run.
    policy : str
        Name of the policy that selected this seed.
    traj_id : int
        Source trajectory index.
    frame_id : int
        Frame index within the source trajectory.
    cluster_id : int
        Cluster from which the seed was drawn.
    global_index : int
        Row index in the concatenated feature matrix.
    """

    seed_id: int
    policy: str
    traj_id: int
    frame_id: int
    cluster_id: int
    global_index: int
