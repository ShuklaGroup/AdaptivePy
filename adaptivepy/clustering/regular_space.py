"""Regular-space clustering for molecular dynamics feature data.

Frames are assigned to clusters such that each cluster center is at least
``min_dist`` away from all previously selected centers (in feature space).
Remaining frames are assigned to the nearest center.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from adaptivepy.clustering.base import Clusterer


class SklearnRegularSpaceClusterer(Clusterer):
    """Greedy regular-space clustering in feature space.

    This implements a distance-threshold variant commonly used in MD analysis:
    cluster seeds are chosen iteratively so that no two centers are closer than
    ``min_dist``. All frames are then assigned to their nearest center.

    Parameters
    ----------
    min_dist : float
        Minimum Euclidean distance between cluster centers.
    max_clusters : int or None
        Optional upper bound on the number of clusters. If ``None``, clustering
        continues until no frame is farther than ``min_dist`` from existing
        centers.
    random_state : int or None
        Seed for shuffling frame order when selecting new centers.
    """

    def __init__(
        self,
        min_dist: float,
        max_clusters: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if min_dist <= 0:
            raise ValueError("min_dist must be positive.")
        self.min_dist = min_dist
        self.max_clusters = max_clusters
        self.random_state = random_state
        self._centers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SklearnRegularSpaceClusterer":
        """Select regular-space centers and assign all frames.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        SklearnRegularSpaceClusterer
            Fitted clusterer.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            self._centers = np.empty((0, X.shape[1]))
            self._labels = np.empty((0,), dtype=int)
            return self

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n_samples)

        center_indices: list[int] = []
        for idx in order:
            if self.max_clusters is not None and len(center_indices) >= self.max_clusters:
                break
            if not center_indices:
                center_indices.append(int(idx))
                continue
            dists = pairwise_distances(
                X[idx : idx + 1], X[center_indices], metric="euclidean"
            )[0]
            if np.all(dists >= self.min_dist):
                center_indices.append(int(idx))

        self._centers = X[center_indices].copy()
        if len(center_indices) == 1:
            self._labels = np.zeros(n_samples, dtype=int)
        else:
            dist_matrix = pairwise_distances(X, self._centers, metric="euclidean")
            self._labels = np.argmin(dist_matrix, axis=1).astype(int)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign labels by nearest regular-space center.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Cluster labels.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if self._centers is None or self._labels is None:
            raise RuntimeError("Clusterer must be fitted before predict.")
        if self._centers.shape[0] == 0:
            return np.zeros(X.shape[0], dtype=int)
        dist_matrix = pairwise_distances(X, self._centers, metric="euclidean")
        return np.argmin(dist_matrix, axis=1).astype(int)

    @property
    def cluster_centers_(self) -> Optional[np.ndarray]:
        """Return regular-space cluster centers."""
        return self._centers

    @property
    def model(self) -> Any:
        """Return a dict representation of the fitted regular-space model."""
        if self._centers is None:
            raise RuntimeError("Clusterer must be fitted before accessing model.")
        return {
            "centers": self._centers,
            "labels": self._labels,
            "min_dist": self.min_dist,
            "max_clusters": self.max_clusters,
        }
