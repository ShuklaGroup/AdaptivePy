"""Least-counts adaptive sampling policy."""

from __future__ import annotations

from typing import List

from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats, sort_clusters_by_population


@register_policy
class LeastCountsPolicy(Policy):
    """Select clusters with the smallest populations.

    Clusters are sorted by ascending population and the first ``n_seeds``
    cluster IDs are returned (one seed per cluster).
    """

    name = "least_counts"

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Select the least-populated clusters.

        Parameters
        ----------
        cluster_stats : dict
            Per-cluster statistics.
        n_seeds : int
            Number of clusters to select.

        Returns
        -------
        list of int
            Cluster IDs with smallest populations.
        """
        sorted_clusters = sort_clusters_by_population(
            cluster_stats, ascending=True
        )
        return sorted_clusters[:n_seeds]
