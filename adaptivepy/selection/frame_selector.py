"""Frame-level seed selection within chosen clusters."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from adaptivepy.models import FrameRecord, SeedResult
from adaptivepy.stats.cluster_stats import ClusterStats


def _nearest_center_frame(
    frames: List[FrameRecord],
    center: np.ndarray,
) -> FrameRecord:
    """Return the frame closest to a cluster centroid in feature space.

    Parameters
    ----------
    frames : list of FrameRecord
        Frames belonging to one cluster.
    center : np.ndarray
        Cluster center, shape ``(n_features,)``.

    Returns
    -------
    FrameRecord
        Frame with minimum Euclidean distance to ``center``.
    """
    features = np.stack([f.features for f in frames], axis=0)
    dists = np.linalg.norm(features - center, axis=1)
    return frames[int(np.argmin(dists))]


def _random_frame(
    frames: List[FrameRecord],
    rng: np.random.Generator,
) -> FrameRecord:
    """Return a uniformly random frame from a cluster.

    Parameters
    ----------
    frames : list of FrameRecord
        Frames belonging to one cluster.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    FrameRecord
        Randomly selected frame.
    """
    index = int(rng.integers(0, len(frames)))
    return frames[index]


def select_seeds(
    policy_name: str,
    selected_clusters: List[int],
    cluster_stats: ClusterStats,
    cluster_centers: Optional[np.ndarray],
    method: str = "nearest_center",
    random_state: Optional[int] = None,
) -> List[SeedResult]:
    """Select one seed frame from each chosen cluster.

    Parameters
    ----------
    policy_name : str
        Name of the policy that selected the clusters.
    selected_clusters : list of int
        Cluster IDs chosen by the policy.
    cluster_stats : dict
        Per-cluster frame lists and populations.
    cluster_centers : np.ndarray or None
        Cluster centroids, shape ``(n_clusters, n_features)``. Required for
        ``nearest_center`` selection when centers are defined per label index.
    method : str
        Selection method: ``nearest_center`` or ``random_frame``.
    random_state : int or None
        Random seed for ``random_frame`` selection.

    Returns
    -------
    list of SeedResult
        Selected seed frames with metadata.

    Raises
    ------
    ValueError
        If ``method`` is unknown or centers are missing when required.
    """
    if method not in {"nearest_center", "random_frame"}:
        raise ValueError(
            f"Unknown seed selection method '{method}'. "
            "Use 'nearest_center' or 'random_frame'."
        )

    rng = np.random.default_rng(random_state)
    seeds: List[SeedResult] = []

    for seed_id, cluster_id in enumerate(selected_clusters):
        entry = cluster_stats.get(cluster_id)
        if entry is None or not entry["frames"]:
            continue

        frames = entry["frames"]

        if method == "random_frame":
            chosen = _random_frame(frames, rng)
        else:
            if cluster_centers is None:
                center = np.mean(np.stack([f.features for f in frames]), axis=0)
            elif cluster_id < len(cluster_centers):
                center = cluster_centers[cluster_id]
            else:
                center = np.mean(np.stack([f.features for f in frames]), axis=0)
            chosen = _nearest_center_frame(frames, center)

        seeds.append(
            SeedResult(
                seed_id=seed_id,
                policy=policy_name,
                traj_id=chosen.traj_id,
                frame_id=chosen.frame_id,
                cluster_id=cluster_id,
                global_index=chosen.global_index or 0,
            )
        )

    return seeds
