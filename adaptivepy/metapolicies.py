"""Cluster-level ensemble strategies for adaptive sampling policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adaptivepy.models import Dataset
from adaptivepy.policies.base import Policy
from adaptivepy.stats.cluster_stats import ClusterStats


@dataclass
class PolicyClusterRanking:
    """A policy's ordered cluster preferences for metapolicy aggregation."""

    policy: str
    ranked_clusters: List[int]
    scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class MetapolicyDecision:
    """Selected ensemble clusters and audit rows."""

    selected_clusters: List[int]
    vote_rows: List[Dict[str, Any]]


def frame_score_cluster_ranking(
    policy_name: str,
    scores: Dict[int, Dict[str, Any]],
    dataset: Dataset,
    cluster_stats: ClusterStats,
    rank_depth: int,
    score_key: str,
) -> PolicyClusterRanking:
    """Aggregate frame-level policy scores to cluster-level maximum score."""
    global_to_cluster = {
        int(frame.global_index): frame.cluster_id
        for frame in dataset.frames
        if frame.global_index is not None and frame.cluster_id is not None
    }
    max_scores: Dict[int, float] = {}
    for global_index, row in scores.items():
        cluster_id = global_to_cluster.get(int(global_index))
        if cluster_id is None:
            continue
        score = float(row[score_key])
        current = max_scores.get(cluster_id)
        if current is None or score > current:
            max_scores[cluster_id] = score

    ranked = sorted(
        max_scores,
        key=lambda cid: (
            -max_scores[cid],
            int(cluster_stats[cid]["population"]),
            cid,
        ),
    )
    return PolicyClusterRanking(
        policy=policy_name,
        ranked_clusters=ranked[:rank_depth],
        scores=max_scores,
    )


def maxent_cluster_ranking(
    policy_name: str,
    scores: Dict[int, Dict[str, Any]],
    dataset: Dataset,
    cluster_stats: ClusterStats,
    rank_depth: int,
) -> PolicyClusterRanking:
    """Aggregate MaxEnt frame entropy to cluster-level maximum entropy."""
    return frame_score_cluster_ranking(
        policy_name=policy_name,
        scores=scores,
        dataset=dataset,
        cluster_stats=cluster_stats,
        rank_depth=rank_depth,
        score_key="entropy",
    )


def ts_dar_cluster_ranking(
    policy_name: str,
    scores: Dict[int, Dict[str, Any]],
    dataset: Dataset,
    cluster_stats: ClusterStats,
    rank_depth: int,
) -> PolicyClusterRanking:
    """Aggregate TS-DAR frame OOD scores to cluster-level maximum OOD score."""
    return frame_score_cluster_ranking(
        policy_name=policy_name,
        scores=scores,
        dataset=dataset,
        cluster_stats=cluster_stats,
        rank_depth=rank_depth,
        score_key="ood_score",
    )


def cluster_policy_ranking(
    policy_name: str,
    policy: Policy,
    cluster_stats: ClusterStats,
    rank_depth: int,
) -> PolicyClusterRanking:
    """Return a cluster policy's ranked cluster IDs."""
    ranked = policy.select_clusters(cluster_stats, rank_depth)
    scores = {
        cluster_id: float(rank_depth - rank)
        for rank, cluster_id in enumerate(ranked)
    }
    return PolicyClusterRanking(
        policy=policy_name,
        ranked_clusters=ranked,
        scores=scores,
    )


def _rank_maps(
    rankings: List[PolicyClusterRanking],
) -> Dict[str, Dict[int, int]]:
    return {
        ranking.policy: {
            cluster_id: rank + 1
            for rank, cluster_id in enumerate(ranking.ranked_clusters)
        }
        for ranking in rankings
    }


def _audit_rows(
    rankings: List[PolicyClusterRanking],
    cluster_stats: ClusterStats,
    selected_clusters: List[int],
    rank_depth: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    weights = weights or {}
    selected = set(selected_clusters)
    rank_by_policy = _rank_maps(rankings)
    rows: List[Dict[str, Any]] = []

    for cluster_id in sorted(cluster_stats):
        vote_count = 0
        ensemble_score = 0.0
        row: Dict[str, Any] = {
            "cluster_id": cluster_id,
            "selected": cluster_id in selected,
            "population": int(cluster_stats[cluster_id]["population"]),
        }
        for ranking in rankings:
            rank = rank_by_policy[ranking.policy].get(cluster_id)
            row[f"rank_{ranking.policy}"] = "" if rank is None else rank
            score = ranking.scores.get(cluster_id)
            row[f"score_{ranking.policy}"] = "" if score is None else score
            if rank is not None:
                vote_count += 1
                ensemble_score += float(weights.get(ranking.policy, 1.0)) * (
                    rank_depth - (rank - 1)
                )
        row["vote_count"] = vote_count
        row["ensemble_score"] = ensemble_score
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            -int(row["selected"]),
            -int(row["vote_count"]),
            -float(row["ensemble_score"]),
            int(row["population"]),
            int(row["cluster_id"]),
        ),
    )


def majority_polling(
    rankings: List[PolicyClusterRanking],
    cluster_stats: ClusterStats,
    n_seeds: int,
    rank_depth: int,
    weights: Optional[Dict[str, float]] = None,
) -> MetapolicyDecision:
    """Select clusters by consensus vote count and weighted rank score."""
    rows = _audit_rows(rankings, cluster_stats, [], rank_depth, weights)
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -int(row["vote_count"]),
            -float(row["ensemble_score"]),
            int(row["population"]),
            int(row["cluster_id"]),
        ),
    )
    n_select = min(int(n_seeds), len(cluster_stats))
    selected_clusters = [int(row["cluster_id"]) for row in ranked_rows[:n_select]]
    return MetapolicyDecision(
        selected_clusters=selected_clusters,
        vote_rows=_audit_rows(
            rankings, cluster_stats, selected_clusters, rank_depth, weights
        ),
    )


def allocation(
    rankings: List[PolicyClusterRanking],
    cluster_stats: ClusterStats,
    allocations: Dict[str, int],
    n_seeds: int,
    rank_depth: int,
    weights: Optional[Dict[str, float]] = None,
) -> MetapolicyDecision:
    """Select clusters by per-policy quotas, skipping duplicates."""
    selected: List[int] = []
    selected_set: set[int] = set()

    for ranking in rankings:
        quota = int(allocations[ranking.policy])
        filled = 0
        for cluster_id in ranking.ranked_clusters:
            if cluster_id in selected_set:
                continue
            selected.append(cluster_id)
            selected_set.add(cluster_id)
            filled += 1
            if filled == quota:
                break

    target = min(int(n_seeds), len(cluster_stats))
    if len(selected) < target:
        fallback = majority_polling(
            rankings=rankings,
            cluster_stats=cluster_stats,
            n_seeds=len(cluster_stats),
            rank_depth=rank_depth,
            weights=weights,
        ).selected_clusters
        for cluster_id in fallback:
            if cluster_id in selected_set:
                continue
            selected.append(cluster_id)
            selected_set.add(cluster_id)
            if len(selected) == target:
                break

    selected = selected[:target]
    return MetapolicyDecision(
        selected_clusters=selected,
        vote_rows=_audit_rows(rankings, cluster_stats, selected, rank_depth, weights),
    )
