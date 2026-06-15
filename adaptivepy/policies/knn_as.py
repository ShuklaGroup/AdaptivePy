"""k-nearest neighbors adaptive sampling policy."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from adaptivepy.models import FrameRecord
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats

ScoringMode = Literal["vectorsum", "distance"]


def compute_knn_as_scores(
    vectors: np.ndarray,
    k: int,
    scoring: ScoringMode = "vectorsum",
) -> Tuple[np.ndarray, int]:
    """Compute kNN-AS scores for representative feature vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Representative feature matrix, shape ``(n_states, n_features)``.
    k : int
        Number of nearest-neighbor records requested from scikit-learn. This
        includes the point itself when present, matching the upstream algorithm.
    scoring : str
        ``vectorsum`` for the magnitude of summed neighbor displacement vectors,
        or ``distance`` for mean Euclidean neighbor distance.

    Returns
    -------
    tuple
        ``(scores, effective_k)`` where ``scores`` has one value per row and
        ``effective_k`` is the clamped neighbor count used for fitting.
    """
    if k < 2:
        raise ValueError("kNN-AS 'k' must be >= 2.")
    if scoring not in {"vectorsum", "distance"}:
        raise ValueError("kNN-AS 'scoring' must be 'vectorsum' or 'distance'.")

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("kNN-AS vectors must be a 2D array.")

    n_states = vectors.shape[0]
    if n_states == 0:
        return np.array([], dtype=float), 0

    effective_k = min(int(k), n_states)
    knn = NearestNeighbors(n_neighbors=effective_k).fit(vectors)
    distances, indices = knn.kneighbors(vectors)

    scores = np.zeros(n_states, dtype=float)
    for row_idx in range(n_states):
        neighbor_mask = indices[row_idx] != row_idx
        neighbor_indices = indices[row_idx][neighbor_mask]
        neighbor_distances = distances[row_idx][neighbor_mask]

        if len(neighbor_indices) == 0:
            scores[row_idx] = 0.0
        elif scoring == "vectorsum":
            displacement = vectors[neighbor_indices] - vectors[row_idx]
            scores[row_idx] = float(np.linalg.norm(displacement.sum(axis=0)))
        else:
            scores[row_idx] = float(np.mean(neighbor_distances))

    return scores, effective_k


def _mean_feature_vector(frames: Sequence[FrameRecord]) -> np.ndarray:
    """Return the mean feature vector for a non-empty frame sequence."""
    return np.mean(np.stack([frame.features for frame in frames], axis=0), axis=0)


def _representative_vector(
    cluster_id: int,
    frames: Sequence[FrameRecord],
    cluster_centers: Optional[np.ndarray],
) -> np.ndarray:
    """Return a finite representative vector for a cluster."""
    expected_shape = np.asarray(frames[0].features).shape
    if cluster_centers is not None and 0 <= cluster_id < len(cluster_centers):
        center = np.asarray(cluster_centers[cluster_id], dtype=float)
        if (
            center.ndim == 1
            and center.shape == expected_shape
            and np.all(np.isfinite(center))
        ):
            return center
    return _mean_feature_vector(frames)


@register_policy
class KnnAsPolicy(Policy):
    """Select clusters using k-nearest neighbors adaptive sampling.

    The original kNN-AS algorithm ranks states by local-neighborhood geometry.
    AdaptivePy policies select clusters, so this implementation applies the
    same ranking to cluster representative vectors and lets the seed-selection
    layer choose a frame from each selected cluster.

    Parameters
    ----------
    k : int
        Number of nearest-neighbor records requested. Includes the query point
        itself when returned by scikit-learn. Default ``5``.
    scoring : str
        ``vectorsum`` matches the upstream vector-sum magnitude mode, while
        ``distance`` uses mean neighbor distance.
    cluster_centers : np.ndarray or None
        Optional cluster centers, shape ``(n_clusters, n_features)``.
    """

    name = "knn_as"

    def __init__(
        self,
        k: int = 5,
        scoring: ScoringMode = "vectorsum",
        cluster_centers: Optional[np.ndarray] = None,
    ) -> None:
        if int(k) < 2:
            raise ValueError("kNN-AS 'k' must be >= 2.")
        if scoring not in {"vectorsum", "distance"}:
            raise ValueError("kNN-AS 'scoring' must be 'vectorsum' or 'distance'.")

        self.k = int(k)
        self.scoring: ScoringMode = scoring
        self.cluster_centers = cluster_centers
        self.last_scores: Dict[int, Dict[str, Any]] = {}

    def _representatives(
        self,
        cluster_stats: ClusterStats,
    ) -> Tuple[List[int], np.ndarray]:
        """Build representative vectors for populated clusters."""
        cluster_ids: List[int] = []
        vectors: List[np.ndarray] = []

        for cluster_id in sorted(cluster_stats):
            frames = cluster_stats[cluster_id]["frames"]
            if not frames:
                continue
            cluster_ids.append(cluster_id)
            vectors.append(
                _representative_vector(cluster_id, frames, self.cluster_centers)
            )

        if not vectors:
            return [], np.empty((0, 0), dtype=float)
        return cluster_ids, np.stack(vectors, axis=0)

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Select clusters with the highest kNN-AS scores."""
        if not cluster_stats:
            self.last_scores = {}
            return []

        cluster_ids, vectors = self._representatives(cluster_stats)
        if not cluster_ids:
            self.last_scores = {}
            return []

        scores, effective_k = compute_knn_as_scores(
            vectors,
            k=self.k,
            scoring=self.scoring,
        )
        score_by_cluster = {
            cluster_id: float(scores[idx])
            for idx, cluster_id in enumerate(cluster_ids)
        }

        self.last_scores = {
            cluster_id: {
                "score": score_by_cluster[cluster_id],
                "population": int(cluster_stats[cluster_id]["population"]),
                "scoring": self.scoring,
                "effective_k": int(effective_k),
            }
            for cluster_id in cluster_ids
        }

        ranked = sorted(
            cluster_ids,
            key=lambda cid: (
                -score_by_cluster[cid],
                cluster_stats[cid]["population"],
                cid,
            ),
        )
        return ranked[:n_seeds]
