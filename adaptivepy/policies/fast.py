"""FAST (Fluctuation Amplification of Specific Traits) sampling policy."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from adaptivepy.models import FrameRecord
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats

Direction = Literal["maximize", "minimize"]


def feature_scale(
    values: Dict[int, float],
    direction: Direction,
) -> Dict[int, float]:
    """Min-max scale cluster descriptor values to [0, 1].

    Parameters
    ----------
    values : dict
        Mapping from cluster ID to raw descriptor value.
    direction : str
        ``maximize`` or ``minimize``.

    Returns
    -------
    dict
        Scaled values in [0, 1]. Returns zeros when all values are equal.
    """
    if not values:
        return {}

    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {cluster_id: 0.0 for cluster_id in values}

    scaled: Dict[int, float] = {}
    for cluster_id, value in values.items():
        if direction == "maximize":
            scaled[cluster_id] = (value - vmin) / (vmax - vmin)
        else:
            scaled[cluster_id] = (vmax - value) / (vmax - vmin)
    return scaled


def compute_exploration_scores(cluster_stats: ClusterStats) -> Dict[int, float]:
    """Compute exploration scores favoring poorly sampled clusters.

    Uses the paper's least-counts component:
    ``(Cmax - C_i) / (Cmax - Cmin)``.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics.

    Returns
    -------
    dict
        Exploration scores in [0, 1] per cluster ID.
    """
    populations = {
        cluster_id: entry["population"]
        for cluster_id, entry in cluster_stats.items()
    }
    return feature_scale(populations, direction="minimize")


def compute_cluster_descriptor(
    frames: List[FrameRecord],
    feature_index: int,
) -> float:
    """Compute mean feature value for a cluster.

    Parameters
    ----------
    frames : list of FrameRecord
        Frames belonging to one cluster.
    feature_index : int
        Column index in the feature vector.

    Returns
    -------
    float
        Mean feature value across cluster frames.
    """
    values = np.array([frame.features[feature_index] for frame in frames], dtype=float)
    return float(np.mean(values))


def compute_directed_scores(
    cluster_stats: ClusterStats,
    feature_indices: Sequence[int],
    directions: Sequence[Direction],
    weights: Sequence[float],
) -> Dict[int, float]:
    """Compute weighted directed exploitation scores across selected features.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics.
    feature_indices : sequence of int
        Feature column indices to use.
    directions : sequence of str
        Optimization direction per feature.
    weights : sequence of float
        Non-negative weights per feature (need not sum to 1).

    Returns
    -------
    dict
        Combined directed score per cluster ID in [0, 1].
    """
    cluster_ids = list(cluster_stats.keys())
    if not cluster_ids:
        return {}

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("FAST weights must sum to a positive value.")

    combined = {cluster_id: 0.0 for cluster_id in cluster_ids}

    for feature_index, direction, weight in zip(feature_indices, directions, weights):
        raw_values = {
            cluster_id: compute_cluster_descriptor(entry["frames"], feature_index)
            for cluster_id, entry in cluster_stats.items()
        }
        scaled = feature_scale(raw_values, direction)
        for cluster_id in cluster_ids:
            combined[cluster_id] += (weight / total_weight) * scaled[cluster_id]

    return combined


def compute_fast_rewards(
    cluster_stats: ClusterStats,
    feature_indices: Sequence[int],
    directions: Sequence[Direction],
    weights: Sequence[float],
    alpha: float,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Compute FAST reward components for all clusters.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics.
    feature_indices : sequence of int
        Feature column indices to use.
    directions : sequence of str
        Optimization direction per feature.
    weights : sequence of float
        Weights per feature.
    alpha : float
        Exploration/exploitation balance parameter.

    Returns
    -------
    tuple of dict
        ``(directed_scores, exploration_scores, rewards)`` keyed by cluster ID.
    """
    directed = compute_directed_scores(
        cluster_stats, feature_indices, directions, weights
    )
    exploration = compute_exploration_scores(cluster_stats)
    rewards = {
        cluster_id: directed[cluster_id] + alpha * exploration[cluster_id]
        for cluster_id in directed
    }
    return directed, exploration, rewards


@register_policy
class FastPolicy(Policy):
    """Select clusters by balancing feature-directed exploitation and exploration.

    Implements the FAST reward from Zimmerman & Bowman (2015):
    ``r(i) = phi_bar(i) + alpha * psi_bar(i)``, where ``phi_bar`` is a
    feature-scaled directed component and ``psi_bar`` favors poorly sampled
    clusters.

    Parameters
    ----------
    feature_indices : sequence of int
        Feature column indices to optimize (required).
    directions : sequence of str or None
        ``maximize`` or ``minimize`` per feature. Defaults to all ``maximize``.
    weights : sequence of float or None
        Weights per feature. Defaults to equal weights.
    alpha : float
        Relative weight of the exploration term. Default ``1.0``.
    """

    name = "fast"

    def __init__(
        self,
        feature_indices: Sequence[int],
        directions: Optional[Sequence[Direction]] = None,
        weights: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
    ) -> None:
        if not feature_indices:
            raise ValueError("FAST policy requires at least one feature index.")

        self.feature_indices = [int(i) for i in feature_indices]
        self.directions: List[Direction] = (
            list(directions)
            if directions is not None
            else ["maximize"] * len(self.feature_indices)
        )
        if len(self.directions) != len(self.feature_indices):
            raise ValueError(
                "FAST directions must match the number of feature indices."
            )
        for direction in self.directions:
            if direction not in {"maximize", "minimize"}:
                raise ValueError(
                    f"Invalid FAST direction '{direction}'. "
                    "Use 'maximize' or 'minimize'."
                )

        if weights is None:
            self.weights = [1.0] * len(self.feature_indices)
        else:
            self.weights = [float(w) for w in weights]
        if len(self.weights) != len(self.feature_indices):
            raise ValueError("FAST weights must match the number of feature indices.")
        if any(w < 0 for w in self.weights):
            raise ValueError("FAST weights must be non-negative.")
        if sum(self.weights) <= 0:
            raise ValueError("FAST weights must sum to a positive value.")

        if alpha < 0:
            raise ValueError("FAST alpha must be non-negative.")
        self.alpha = float(alpha)

        self.last_scores: Dict[int, Dict[str, float]] = {}

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Select clusters with the highest FAST reward scores.

        Parameters
        ----------
        cluster_stats : dict
            Per-cluster statistics.
        n_seeds : int
            Number of clusters to select.

        Returns
        -------
        list of int
            Cluster IDs with highest rewards.
        """
        if not cluster_stats:
            self.last_scores = {}
            return []

        directed, exploration, rewards = compute_fast_rewards(
            cluster_stats,
            self.feature_indices,
            self.directions,
            self.weights,
            self.alpha,
        )

        self.last_scores = {
            cluster_id: {
                "directed_score": directed[cluster_id],
                "exploration_score": exploration[cluster_id],
                "reward": rewards[cluster_id],
                "population": float(cluster_stats[cluster_id]["population"]),
            }
            for cluster_id in rewards
        }

        ranked = sorted(
            rewards.keys(),
            key=lambda cid: (
                -rewards[cid],
                cluster_stats[cid]["population"],
                cid,
            ),
        )
        return ranked[:n_seeds]
