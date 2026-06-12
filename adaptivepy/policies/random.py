"""Random cluster selection policy."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats


@register_policy
class RandomPolicy(Policy):
    """Uniformly sample cluster IDs at random.

    Parameters
    ----------
    random_state : int or None
        Seed for the random number generator.
    """

    name = "random"

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Randomly sample ``n_seeds`` distinct cluster IDs.

        Parameters
        ----------
        cluster_stats : dict
            Per-cluster statistics.
        n_seeds : int
            Number of clusters to sample.

        Returns
        -------
        list of int
            Randomly selected cluster IDs.
        """
        cluster_ids = list(cluster_stats.keys())
        n_select = min(n_seeds, len(cluster_ids))
        if n_select == 0:
            return []
        chosen = self._rng.choice(cluster_ids, size=n_select, replace=False)
        return [int(c) for c in chosen]
