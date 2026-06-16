"""Adaptive sampling policy base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

from adaptivepy.models import Dataset, SeedResult
from adaptivepy.stats.cluster_stats import ClusterStats

POLICY_REGISTRY: Dict[str, Type["Policy"]] = {}


def register_policy(cls: Type["Policy"]) -> Type["Policy"]:
    """Register a policy class in :data:`POLICY_REGISTRY`.

    Parameters
    ----------
    cls : type
        Policy subclass with a ``name`` class attribute.

    Returns
    -------
    type
        The registered policy class (unchanged).

    Raises
    ------
    ValueError
        If the policy name is missing or already registered.
    """
    if not getattr(cls, "name", None):
        raise ValueError(f"Policy {cls.__name__} must define a 'name' attribute.")
    if cls.name in POLICY_REGISTRY:
        raise ValueError(f"Policy '{cls.name}' is already registered.")
    POLICY_REGISTRY[cls.name] = cls
    return cls


class Policy(ABC):
    """Base class for adaptive sampling policies.

    Cluster-based policies implement :meth:`select_clusters` and set
    ``requires_clustering = True``. Frame-level policies set
    ``requires_clustering = False`` and implement :meth:`select_frames`.
    """

    name: str = ""
    requires_clustering: bool = True

    @abstractmethod
    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Select cluster IDs from which to draw seed frames.

        Parameters
        ----------
        cluster_stats : dict
            Per-cluster population and frame lists.
        n_seeds : int
            Maximum number of clusters (seeds) to select.

        Returns
        -------
        list of int
            Selected cluster IDs.
        """
        ...

    def select_frames(
        self,
        dataset: Dataset,
        n_seeds: int,
    ) -> List[SeedResult]:
        """Select seed frames directly from loaded features.

        Frame-level policies override this method. Cluster-based policies
        raise :exc:`NotImplementedError`.

        Parameters
        ----------
        dataset : Dataset
            Loaded feature dataset.
        n_seeds : int
            Number of seed frames to select.

        Returns
        -------
        list of SeedResult
            Selected seed frames.
        """
        raise NotImplementedError(
            f"Policy '{self.name}' does not support direct frame selection."
        )


def policy_requires_clustering(name: str) -> bool:
    """Return whether a registered policy needs clustering.

    Parameters
    ----------
    name : str
        Registered policy name.

    Returns
    -------
    bool
        ``True`` when the policy consumes cluster statistics.
    """
    if name not in POLICY_REGISTRY:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    return bool(getattr(POLICY_REGISTRY[name], "requires_clustering", True))


def get_policy(name: str, **kwargs) -> Policy:
    """Instantiate a registered policy by name.

    Parameters
    ----------
    name : str
        Registered policy name.
    **kwargs
        Constructor arguments forwarded to the policy class.

    Returns
    -------
    Policy
        Policy instance.

    Raises
    ------
    ValueError
        If the policy name is unknown.
    """
    if name not in POLICY_REGISTRY:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    return POLICY_REGISTRY[name](**kwargs)


def list_policies() -> List[str]:
    """Return names of all registered policies.

    Returns
    -------
    list of str
        Sorted policy names.
    """
    return sorted(POLICY_REGISTRY.keys())
