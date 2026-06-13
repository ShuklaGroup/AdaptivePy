"""Multiagent REAP (MA-REAP) adaptive sampling policy."""

from __future__ import annotations

import sys
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize

from adaptivepy.models import FrameRecord
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats, sort_clusters_by_population

StakesMethod = Literal["percentage", "equal", "max", "logistic"]
Regime = Literal["collaborative", "noncollaborative", "competitive"]

_EPSILON = sys.float_info.epsilon


def build_traj_id_to_agent(
    agent_assignments: Dict[str, Sequence[str]],
    traj_names: Sequence[str],
) -> Dict[int, str]:
    """Map trajectory index to agent name from stem assignments."""
    stem_to_id = {name: idx for idx, name in enumerate(traj_names)}
    mapping: Dict[int, str] = {}
    for agent_name, stems in agent_assignments.items():
        for stem in stems:
            if stem not in stem_to_id:
                raise ValueError(
                    f"MA-REAP agent '{agent_name}' references unknown trajectory '{stem}'."
                )
            traj_id = stem_to_id[stem]
            if traj_id in mapping:
                raise ValueError(
                    f"Trajectory '{stem}' is assigned to multiple MA-REAP agents."
                )
            mapping[traj_id] = agent_name
    return mapping


def select_least_count_candidates(
    cluster_stats: ClusterStats,
    n_candidates: int,
) -> List[int]:
    """Return least-populated cluster IDs as MA-REAP action candidates."""
    sorted_ids = sort_clusters_by_population(cluster_stats, ascending=True)
    return sorted_ids[: min(n_candidates, len(sorted_ids))]


def candidate_feature_vectors(
    cluster_stats: ClusterStats,
    candidate_ids: Sequence[int],
    cluster_centers: Optional[np.ndarray],
) -> np.ndarray:
    """Feature vectors for candidate clusters, shape ``(n_candidates, n_features)``."""
    vectors: List[np.ndarray] = []
    for cluster_id in candidate_ids:
        frames = cluster_stats[cluster_id]["frames"]
        if cluster_centers is not None and 0 <= cluster_id < len(cluster_centers):
            vectors.append(np.asarray(cluster_centers[cluster_id], dtype=float))
        else:
            vectors.append(
                np.mean(np.stack([f.features for f in frames], axis=0), axis=0)
            )
    return np.stack(vectors, axis=0)


def count_agent_frames_in_clusters(
    cluster_stats: ClusterStats,
    candidate_ids: Sequence[int],
    agent_names: Sequence[str],
    traj_id_to_agent: Dict[int, str],
) -> np.ndarray:
    """Count frames per agent in each candidate cluster.

    Returns
    -------
    np.ndarray
        Shape ``(n_agents, n_candidates)``.
    """
    agent_index = {name: idx for idx, name in enumerate(agent_names)}
    counts = np.zeros((len(agent_names), len(candidate_ids)), dtype=float)
    for col, cluster_id in enumerate(candidate_ids):
        for frame in cluster_stats[cluster_id]["frames"]:
            agent = traj_id_to_agent.get(frame.traj_id)
            if agent is not None:
                counts[agent_index[agent], col] += 1.0
    return counts


def apply_stakes_method(
    raw_counts: np.ndarray,
    method: StakesMethod,
    stakes_k: Optional[float] = None,
) -> np.ndarray:
    """Convert raw frame counts to normalized stakes per candidate column."""
    stakes = raw_counts.copy()
    n_agents, n_candidates = stakes.shape

    if method == "percentage":
        return normalize(stakes, norm="l1", axis=0)

    for col in range(n_candidates):
        nonzero = stakes[:, col] != 0
        if not np.any(nonzero):
            continue
        if method == "equal":
            stakes[nonzero, col] = 1.0 / np.count_nonzero(stakes[:, col])
        elif method == "max":
            stakes[:, col] = 0.0
            stakes[np.argmax(raw_counts[:, col]), col] = 1.0
        elif method == "logistic":
            if stakes_k is None:
                raise ValueError("MA-REAP 'stakes_k' is required for logistic stakes.")
            k = stakes_k
            x0 = 0.5

            def logistic(x: float) -> float:
                return 1.0 / (1.0 + np.exp(-k * (x - x0)))

            col_vals = raw_counts[nonzero, col]
            transformed = np.array([logistic(x) for x in col_vals], dtype=float)
            total = transformed.sum()
            if total > 0:
                stakes[nonzero, col] = transformed / total
        else:
            raise ValueError(f"Unknown MA-REAP stakes method '{method}'.")

    return stakes


def agent_feature_statistics(
    frames: List[FrameRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of features for an agent's frames."""
    if not frames:
        raise ValueError("Cannot compute agent statistics from empty frame list.")
    features = np.stack([f.features for f in frames], axis=0)
    return features.mean(axis=0), features.std(axis=0)


def compute_agent_scores(
    means: np.ndarray,
    stdev: np.ndarray,
    stakes_agent: np.ndarray,
    candidate_features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-candidate reward for one agent (eq. 2 in Kleiman & Shukla 2022)."""
    dist = np.abs(candidate_features - means)
    distances = (weights * dist / (stdev + _EPSILON)).sum(axis=1)
    return stakes_agent * distances


def optimize_agent_weights(
    weights_prev: np.ndarray,
    delta: float,
    means: np.ndarray,
    stdev: np.ndarray,
    stakes_agent: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    """Update CV weights for one agent via SLSQP (mareap_sim.set_weights)."""
    constraints = [
        {
            "type": "ineq",
            "fun": lambda w, wp=weights_prev, d=delta: d - np.abs(wp - w),
            "jac": lambda w, wp=weights_prev, d=delta: np.diagflat(
                np.sign(wp - w)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda w: w,
            "jac": lambda w: np.eye(w.shape[0]),
        },
        {
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones(w.shape[0]),
        },
    ]

    def objective(x: np.ndarray) -> float:
        return -float(
            compute_agent_scores(
                means, stdev, stakes_agent, candidate_features, x
            ).sum()
        )

    result = minimize(
        objective,
        weights_prev,
        method="SLSQP",
        constraints=constraints,
    )
    return np.asarray(result.x, dtype=float)


def aggregate_agent_scores(
    scores: np.ndarray,
    regime: Regime,
) -> np.ndarray:
    """Combine per-agent scores into global candidate scores (eqs. 7-9)."""
    if regime == "collaborative":
        return scores.sum(axis=0)
    if regime == "noncollaborative":
        return scores.max(axis=0)
    if regime == "competitive":
        return 2.0 * scores.max(axis=0) - scores.sum(axis=0)
    raise ValueError(f"Unknown MA-REAP regime '{regime}'.")


def select_executors(stakes: np.ndarray, selected_indices: Sequence[int]) -> List[int]:
    """Return agent index with highest stake for each selected candidate."""
    return [int(idx) for idx in np.argmax(stakes, axis=0)[list(selected_indices)]]


@register_policy
class MaReapPolicy(Policy):
    """Multiagent REAP cluster selection policy.

    Implements Kleiman & Shukla (2022): least-counts candidates, per-agent
    stakes, learned CV weights, and multiagent reward aggregation.

    Parameters
    ----------
    agent_assignments : dict
        Maps agent names to feature file stems.
    traj_names : list of str
        Ordered trajectory stems from the loaded dataset.
    cluster_centers : np.ndarray or None
        Cluster centroids, shape ``(n_clusters, n_features)``.
    n_candidates : int
        Number of least-count clusters to consider.
    initial_weights : np.ndarray or None
        Starting CV weights per agent or shared across agents.
    delta : float
        Maximum per-feature weight change per round.
    stakes_method : str
        ``percentage``, ``equal``, ``max``, or ``logistic``.
    stakes_k : float or None
        Logistic steepness when ``stakes_method='logistic'``.
    regime : str
        ``collaborative``, ``noncollaborative``, or ``competitive``.
    """

    name = "ma_reap"

    def __init__(
        self,
        agent_assignments: Dict[str, Sequence[str]],
        traj_names: Sequence[str],
        cluster_centers: Optional[np.ndarray] = None,
        n_candidates: int = 10,
        initial_weights: Optional[np.ndarray] = None,
        delta: float = 0.05,
        stakes_method: StakesMethod = "percentage",
        stakes_k: Optional[float] = None,
        regime: Regime = "collaborative",
    ) -> None:
        if len(agent_assignments) < 2:
            raise ValueError("MA-REAP requires at least two agents.")

        self.agent_names = sorted(agent_assignments.keys())
        self.traj_id_to_agent = build_traj_id_to_agent(
            agent_assignments, traj_names
        )
        assigned_trajs = set(self.traj_id_to_agent.keys())
        all_trajs = set(range(len(traj_names)))
        missing = all_trajs - assigned_trajs
        if missing:
            unassigned = [traj_names[i] for i in sorted(missing)]
            raise ValueError(
                "MA-REAP requires every trajectory to be assigned to an agent. "
                f"Unassigned: {unassigned}"
            )

        self.cluster_centers = cluster_centers
        self.n_candidates = int(n_candidates)
        self.delta = float(delta)
        self.stakes_method = stakes_method
        self.stakes_k = stakes_k
        self.regime = regime
        self.n_features: Optional[int] = None
        if initial_weights is not None:
            self.initial_weights = np.asarray(initial_weights, dtype=float)
        else:
            self.initial_weights = None

        self.last_scores: Dict[int, Dict[str, float]] = {}
        self.last_weights: Dict[str, List[float]] = {}
        self.last_stakes: Dict[int, Dict[str, float]] = {}
        self.last_executors: Dict[int, str] = {}
        self._agent_scores_matrix: Optional[np.ndarray] = None
        self._candidate_ids: List[int] = []

    def _agent_frames(
        self,
        cluster_stats: ClusterStats,
        agent_name: str,
    ) -> List[FrameRecord]:
        return [
            frame
            for entry in cluster_stats.values()
            for frame in entry["frames"]
            if self.traj_id_to_agent.get(frame.traj_id) == agent_name
        ]

    def _resolve_initial_weights(self, n_features: int, n_agents: int) -> np.ndarray:
        if self.initial_weights is None:
            uniform = np.full(n_features, 1.0 / n_features)
            return np.tile(uniform, (n_agents, 1))
        weights = self.initial_weights
        if weights.ndim == 1 and weights.shape[0] == n_features:
            return np.tile(weights, (n_agents, 1))
        if weights.ndim == 2 and weights.shape == (n_agents, n_features):
            return weights.copy()
        raise ValueError(
            f"MA-REAP initial_weights must be shape ({n_features},) or "
            f"({n_agents}, {n_features}), got {weights.shape}."
        )

    def select_clusters(
        self,
        cluster_stats: ClusterStats,
        n_seeds: int,
    ) -> List[int]:
        """Select clusters using the MA-REAP reward pipeline."""
        if not cluster_stats:
            self.last_scores = {}
            self.last_weights = {}
            self.last_stakes = {}
            self.last_executors = {}
            return []

        candidate_ids = select_least_count_candidates(
            cluster_stats, self.n_candidates
        )
        if not candidate_ids:
            return []

        candidate_features = candidate_feature_vectors(
            cluster_stats, candidate_ids, self.cluster_centers
        )
        n_features = candidate_features.shape[1]
        self.n_features = n_features
        n_agents = len(self.agent_names)

        raw_counts = count_agent_frames_in_clusters(
            cluster_stats,
            candidate_ids,
            self.agent_names,
            self.traj_id_to_agent,
        )
        stakes = apply_stakes_method(
            raw_counts, self.stakes_method, self.stakes_k
        )

        prev_weights = self._resolve_initial_weights(n_features, n_agents)
        new_weights = np.empty_like(prev_weights)
        scores = np.empty((n_agents, len(candidate_ids)))

        for agent_idx, agent_name in enumerate(self.agent_names):
            agent_frames = self._agent_frames(cluster_stats, agent_name)
            means, stdev = agent_feature_statistics(agent_frames)
            stakes_agent = stakes[agent_idx]
            new_weights[agent_idx] = optimize_agent_weights(
                prev_weights[agent_idx],
                self.delta,
                means,
                stdev,
                stakes_agent,
                candidate_features,
            )
            scores[agent_idx] = compute_agent_scores(
                means,
                stdev,
                stakes_agent,
                candidate_features,
                new_weights[agent_idx],
            )

        aggregated = aggregate_agent_scores(scores, self.regime)
        ranked_indices = sorted(
            range(len(candidate_ids)),
            key=lambda idx: (
                -aggregated[idx],
                cluster_stats[candidate_ids[idx]]["population"],
                candidate_ids[idx],
            ),
        )
        selected_local = ranked_indices[: min(n_seeds, len(candidate_ids))]
        selected_cluster_ids = [candidate_ids[i] for i in selected_local]

        self._candidate_ids = candidate_ids
        self._agent_scores_matrix = scores

        self.last_scores = {}
        for col, cluster_id in enumerate(candidate_ids):
            row: Dict[str, float] = {
                "aggregate_score": float(aggregated[col]),
                "population": float(cluster_stats[cluster_id]["population"]),
            }
            for agent_idx, agent_name in enumerate(self.agent_names):
                row[f"score_{agent_name}"] = float(scores[agent_idx, col])
            self.last_scores[cluster_id] = row

        self.last_weights = {
            agent_name: new_weights[idx].tolist()
            for idx, agent_name in enumerate(self.agent_names)
        }

        self.last_stakes = {}
        for col, cluster_id in enumerate(candidate_ids):
            self.last_stakes[cluster_id] = {
                agent_name: float(stakes[agent_idx, col])
                for agent_idx, agent_name in enumerate(self.agent_names)
            }

        executor_indices = np.argmax(stakes, axis=0)[selected_local]
        self.last_executors = {
            selected_cluster_ids[i]: self.agent_names[int(executor_indices[i])]
            for i in range(len(selected_cluster_ids))
        }

        return selected_cluster_ids
