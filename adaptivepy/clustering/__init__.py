"""Clustering backend factory and registry."""

from __future__ import annotations

from typing import Any, Dict, Type

from adaptivepy.clustering.base import Clusterer
from adaptivepy.clustering.regular_space import SklearnRegularSpaceClusterer
from adaptivepy.clustering.sklearn_kmeans import SklearnKMeansClusterer
from adaptivepy.clustering.sklearn_minibatch import SklearnMiniBatchClusterer

CLUSTERER_REGISTRY: Dict[str, Type[Clusterer]] = {
    "kmeans": SklearnKMeansClusterer,
    "minibatch_kmeans": SklearnMiniBatchClusterer,
    "regular_space": SklearnRegularSpaceClusterer,
}


def create_clusterer(
    method: str,
    n_clusters: int,
    random_state: int | None = None,
    params: Dict[str, Any] | None = None,
) -> Clusterer:
    """Instantiate a registered clustering backend.

    Parameters
    ----------
    method : str
        Clustering method name (``kmeans``, ``minibatch_kmeans``,
        ``regular_space``).
    n_clusters : int
        Target number of clusters (used by k-means variants; mapped to
        ``max_clusters`` for regular-space when applicable).
    random_state : int or None
        Random seed for reproducibility.
    params : dict or None
        Additional backend-specific parameters.

    Returns
    -------
    Clusterer
        Unfitted clusterer instance.

    Raises
    ------
    ValueError
        If ``method`` is not registered.
    """
    if method not in CLUSTERER_REGISTRY:
        available = ", ".join(sorted(CLUSTERER_REGISTRY))
        raise ValueError(f"Unknown clustering method '{method}'. Available: {available}")

    params = dict(params or {})
    clusterer_cls = CLUSTERER_REGISTRY[method]

    if method == "regular_space":
        if "min_dist" not in params:
            raise ValueError(
                "regular_space clustering requires 'min_dist' in clustering.params."
            )
        max_clusters = params.pop("max_clusters", n_clusters)
        return SklearnRegularSpaceClusterer(
            min_dist=float(params.pop("min_dist")),
            max_clusters=int(max_clusters) if max_clusters else None,
            random_state=random_state,
            **params,
        )

    return clusterer_cls(
        n_clusters=n_clusters,
        random_state=random_state,
        **params,
    )


def fit_clusterer(clusterer: Clusterer, X) -> Clusterer:
    """Fit a clusterer and return it for chaining.

    Parameters
    ----------
    clusterer : Clusterer
        Unfitted clusterer instance.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    Clusterer
        Fitted clusterer.
    """
    return clusterer.fit(X)


__all__ = [
    "CLUSTERER_REGISTRY",
    "Clusterer",
    "SklearnKMeansClusterer",
    "SklearnMiniBatchClusterer",
    "SklearnRegularSpaceClusterer",
    "create_clusterer",
    "fit_clusterer",
]
