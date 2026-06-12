"""Cluster population statistics and frame assignments."""

from __future__ import annotations

from typing import Dict, List, TypedDict

import numpy as np

from adaptivepy.models import Dataset, FrameRecord


class ClusterStatEntry(TypedDict):
    """Statistics for a single cluster."""

    population: int
    frames: List[FrameRecord]


ClusterStats = Dict[int, ClusterStatEntry]


def assign_clusters(dataset: Dataset, labels: np.ndarray) -> None:
    """Attach cluster labels to frame records in a dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset whose frames will be updated in place.
    labels : np.ndarray
        Cluster label per frame, shape ``(n_frames,)``.

    Raises
    ------
    ValueError
        If label count does not match the number of frames.
    """
    if len(labels) != len(dataset.frames):
        raise ValueError(
            f"Expected {len(dataset.frames)} labels, got {len(labels)}."
        )
    for record, cluster_id in zip(dataset.frames, labels):
        record.cluster_id = int(cluster_id)


def compute_cluster_stats(dataset: Dataset) -> ClusterStats:
    """Compute per-cluster populations and frame lists.

    Parameters
    ----------
    dataset : Dataset
        Dataset with cluster assignments on each frame record.

    Returns
    -------
    dict
        Mapping from ``cluster_id`` to population and frame list.

    Raises
    ------
    ValueError
        If any frame lacks a cluster assignment.
    """
    stats: ClusterStats = {}

    for record in dataset.frames:
        if record.cluster_id is None:
            raise ValueError("All frames must have cluster assignments.")
        cluster_id = record.cluster_id
        if cluster_id not in stats:
            stats[cluster_id] = {"population": 0, "frames": []}
        stats[cluster_id]["population"] += 1
        stats[cluster_id]["frames"].append(record)

    return stats


def sort_clusters_by_population(
    cluster_stats: ClusterStats,
    ascending: bool = True,
) -> List[int]:
    """Return cluster IDs sorted by population.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics from :func:`compute_cluster_stats`.
    ascending : bool
        If ``True``, smallest populations first.

    Returns
    -------
    list of int
        Sorted cluster IDs.
    """
    return sorted(
        cluster_stats.keys(),
        key=lambda cid: cluster_stats[cid]["population"],
        reverse=not ascending,
    )


def cluster_stats_to_rows(cluster_stats: ClusterStats) -> List[Dict[str, int]]:
    """Convert cluster statistics to flat rows for CSV export.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics.

    Returns
    -------
    list of dict
        Rows with keys ``cluster_id`` and ``population``.
    """
    return [
        {"cluster_id": cluster_id, "population": entry["population"]}
        for cluster_id, entry in sorted(cluster_stats.items())
    ]
