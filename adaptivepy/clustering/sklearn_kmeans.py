"""KMeans clustering via scikit-learn."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.cluster import KMeans

from adaptivepy.clustering.base import Clusterer


class SklearnKMeansClusterer(Clusterer):
    """Wrap ``sklearn.cluster.KMeans`` for AdaptivePy.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    random_state : int or None
        Random seed passed to KMeans.
    **kwargs
        Additional keyword arguments forwarded to ``KMeans``.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._extra_kwargs = kwargs
        self._model: Optional[KMeans] = None

    def fit(self, X: np.ndarray) -> "SklearnKMeansClusterer":
        """Fit KMeans on the provided feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        SklearnKMeansClusterer
            Fitted clusterer.
        """
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            **self._extra_kwargs,
        )
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for ``X``.

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
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted before predict.")
        return self._model.predict(X)

    @property
    def cluster_centers_(self) -> Optional[np.ndarray]:
        """Return KMeans cluster centers."""
        if self._model is None:
            return None
        return self._model.cluster_centers_

    @property
    def model(self) -> Any:
        """Return the fitted ``KMeans`` instance."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted before accessing model.")
        return self._model
