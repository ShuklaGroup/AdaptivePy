"""Abstract base class for clustering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class Clusterer(ABC):
    """Base interface for feature-space clustering algorithms.

    Implementations wrap scikit-learn or custom clustering methods and expose
    a uniform ``fit`` / ``predict`` API.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Clusterer":
        """Fit the clusterer to feature data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        Clusterer
            Fitted clusterer instance (``self``).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to feature data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape ``(n_samples,)``.
        """
        ...

    @property
    @abstractmethod
    def cluster_centers_(self) -> Optional[np.ndarray]:
        """Return cluster centroids if available.

        Returns
        -------
        np.ndarray or None
            Array of shape ``(n_clusters, n_features)``, or ``None`` if the
            method does not define explicit centers.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the underlying fitted model object for serialization.

        Returns
        -------
        object
            Backend-specific model instance.
        """
        ...
