"""Write run outputs: CSV metadata, numpy arrays, and serialized models."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import yaml

from adaptivepy.config.schema import RunConfig, config_to_dict
from adaptivepy.models import SeedResult
from adaptivepy.stats.cluster_stats import ClusterStats, cluster_stats_to_rows
from adaptivepy.utils.io_utils import copy_file, ensure_dir

logger = logging.getLogger(__name__)


def write_run_config(config: RunConfig, output_dir: Path, source_path: Path) -> Path:
    """Save a copy of the run configuration to the output directory.

    Parameters
    ----------
    config : RunConfig
        Parsed configuration object.
    output_dir : Path
        Run output directory.
    source_path : Path
        Original YAML file path (copied verbatim when available).

    Returns
    -------
    Path
        Path to the saved configuration file.
    """
    dst = ensure_dir(output_dir) / "run_config.yaml"
    if source_path.is_file():
        copy_file(source_path, dst)
    else:
        with dst.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
    return dst


def write_assignments(assignments: np.ndarray, output_dir: Path) -> Path:
    """Save per-frame cluster assignments as a numpy array.

    Parameters
    ----------
    assignments : np.ndarray
        Cluster label per frame.
    output_dir : Path
        Run output directory.

    Returns
    -------
    Path
        Path to ``assignments.npy``.
    """
    path = ensure_dir(output_dir) / "assignments.npy"
    np.save(path, assignments)
    return path


def write_cluster_model(model: Any, output_dir: Path) -> Path:
    """Serialize the fitted clustering model with joblib.

    Parameters
    ----------
    model : object
        Fitted clustering model.
    output_dir : Path
        Run output directory.

    Returns
    -------
    Path
        Path to ``cluster_model.pkl``.
    """
    path = ensure_dir(output_dir) / "cluster_model.pkl"
    joblib.dump(model, path)
    return path


def write_cluster_statistics(
    cluster_stats: ClusterStats,
    output_dir: Path,
) -> Path:
    """Write cluster population statistics to ``metadata.csv``.

    Parameters
    ----------
    cluster_stats : dict
        Per-cluster statistics.
    output_dir : Path
        Run output directory.

    Returns
    -------
    Path
        Path to ``metadata.csv``.
    """
    path = ensure_dir(output_dir) / "metadata.csv"
    rows = cluster_stats_to_rows(cluster_stats)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "population"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_seeds_csv(seeds: Iterable[SeedResult], output_dir: Path) -> Path:
    """Write selected seeds to ``seeds.csv``.

    Parameters
    ----------
    seeds : iterable of SeedResult
        Seed records for one policy.
    output_dir : Path
        Policy-specific output directory.

    Returns
    -------
    Path
        Path to ``seeds.csv``.
    """
    path = ensure_dir(output_dir) / "seeds.csv"
    fieldnames = [
        "seed_id",
        "policy",
        "traj_id",
        "frame_id",
        "cluster_id",
        "global_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in seeds:
            writer.writerow(
                {
                    "seed_id": seed.seed_id,
                    "policy": seed.policy,
                    "traj_id": seed.traj_id,
                    "frame_id": seed.frame_id,
                    "cluster_id": (
                        "" if seed.cluster_id is None else seed.cluster_id
                    ),
                    "global_index": seed.global_index,
                }
            )
    return path


def write_combined_metadata(
    policy_seeds: Dict[str, List[SeedResult]],
    output_dir: Path,
) -> Path:
    """Write a combined seed table across all policies.

    Parameters
    ----------
    policy_seeds : dict
        Mapping from policy name to seed lists.
    output_dir : Path
        Top-level results directory.

    Returns
    -------
    Path
        Path to ``combined_metadata.csv``.
    """
    path = ensure_dir(output_dir) / "combined_metadata.csv"
    fieldnames = [
        "seed_id",
        "policy",
        "traj_id",
        "frame_id",
        "cluster_id",
        "global_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _policy, seeds in policy_seeds.items():
            for seed in seeds:
                writer.writerow(
                    {
                        "seed_id": seed.seed_id,
                        "policy": seed.policy,
                        "traj_id": seed.traj_id,
                        "frame_id": seed.frame_id,
                        "cluster_id": (
                            "" if seed.cluster_id is None else seed.cluster_id
                        ),
                        "global_index": seed.global_index,
                    }
                )
    logger.info("Wrote combined metadata to %s", path)
    return path


def write_metapolicy_votes(
    vote_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write cluster-level metapolicy vote audit data to ``votes.csv``."""
    path = ensure_dir(output_dir) / "votes.csv"
    base_fields = [
        "cluster_id",
        "selected",
        "vote_count",
        "ensemble_score",
        "population",
    ]
    extra_fields = sorted(
        {
            field
            for row in vote_rows
            for field in row
            if field not in base_fields
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_fields, *extra_fields])
        writer.writeheader()
        for row in vote_rows:
            writer.writerow(row)
    return path


def write_fast_scores(
    scores: Dict[int, Dict[str, float]],
    output_dir: Path,
) -> Path:
    """Write FAST reward components to ``scores.csv``.

    Parameters
    ----------
    scores : dict
        Mapping from cluster ID to score components.
    output_dir : Path
        Policy-specific output directory.

    Returns
    -------
    Path
        Path to ``scores.csv``.
    """
    path = ensure_dir(output_dir) / "scores.csv"
    fieldnames = [
        "cluster_id",
        "directed_score",
        "exploration_score",
        "reward",
        "population",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id in sorted(scores):
            row = scores[cluster_id]
            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "directed_score": row["directed_score"],
                    "exploration_score": row["exploration_score"],
                    "reward": row["reward"],
                    "population": int(row["population"]),
                }
            )
    return path


def write_knn_as_scores(
    scores: Dict[int, Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write kNN-AS scores to ``scores.csv``."""
    path = ensure_dir(output_dir) / "scores.csv"
    fieldnames = ["cluster_id", "score", "population", "scoring", "effective_k"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id in sorted(scores):
            row = scores[cluster_id]
            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "score": row["score"],
                    "population": int(row["population"]),
                    "scoring": row["scoring"],
                    "effective_k": int(row["effective_k"]),
                }
            )
    return path


def write_maxent_vampnet_scores(
    scores: Dict[int, Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write MaxEnt VAMPNet per-frame entropy scores to ``scores.csv``."""
    path = ensure_dir(output_dir) / "scores.csv"
    if not scores:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "global_index",
                    "traj_id",
                    "frame_id",
                    "entropy",
                    "selected",
                ],
            )
            writer.writeheader()
        return path

    n_states = len(next(iter(scores.values()))["probabilities"])
    fieldnames = [
        "global_index",
        "traj_id",
        "frame_id",
        "entropy",
        "selected",
        *[f"prob_{state_idx}" for state_idx in range(n_states)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for global_index in sorted(scores):
            row = scores[global_index]
            output_row: Dict[str, Any] = {
                "global_index": global_index,
                "traj_id": row["traj_id"],
                "frame_id": row["frame_id"],
                "entropy": row["entropy"],
                "selected": bool(row["selected"]),
            }
            for state_idx, prob in enumerate(row["probabilities"]):
                output_row[f"prob_{state_idx}"] = prob
            writer.writerow(output_row)
    return path


def write_ts_dar_scores(
    scores: Dict[int, Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write TS-DAR per-frame OOD scores to ``scores.csv``."""
    path = ensure_dir(output_dir) / "scores.csv"
    if not scores:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "global_index",
                    "traj_id",
                    "frame_id",
                    "ood_score",
                    "state",
                    "selected",
                ],
            )
            writer.writeheader()
        return path

    first_row = next(iter(scores.values()))
    n_embeddings = len(first_row["embedding"])
    n_states = len(first_row["probabilities"])
    fieldnames = [
        "global_index",
        "traj_id",
        "frame_id",
        "ood_score",
        "state",
        "selected",
        *[f"emb_{idx}" for idx in range(n_embeddings)],
        *[f"prob_{idx}" for idx in range(n_states)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for global_index in sorted(scores):
            row = scores[global_index]
            output_row: Dict[str, Any] = {
                "global_index": global_index,
                "traj_id": row["traj_id"],
                "frame_id": row["frame_id"],
                "ood_score": row["ood_score"],
                "state": row["state"],
                "selected": bool(row["selected"]),
            }
            for emb_idx, value in enumerate(row["embedding"]):
                output_row[f"emb_{emb_idx}"] = value
            for prob_idx, prob in enumerate(row["probabilities"]):
                output_row[f"prob_{prob_idx}"] = prob
            writer.writerow(output_row)
    return path


def write_ma_reap_outputs(
    scores: Dict[int, Dict[str, float]],
    weights: Dict[str, List[float]],
    stakes: Dict[int, Dict[str, float]],
    executors: Dict[int, str],
    agent_names: List[str],
    seeds: List[SeedResult],
    output_dir: Path,
) -> None:
    """Write MA-REAP sidecar CSV files under the policy output directory."""
    policy_dir = ensure_dir(output_dir)

    score_fields = ["cluster_id", "aggregate_score", "population"] + [
        f"score_{name}" for name in agent_names
    ]
    scores_path = policy_dir / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        for cluster_id in sorted(scores):
            row = {"cluster_id": cluster_id}
            row.update(scores[cluster_id])
            writer.writerow(row)

    weights_path = policy_dir / "agent_weights.csv"
    with weights_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["agent", "feature_index", "weight"]
        )
        writer.writeheader()
        for agent_name in agent_names:
            for feat_idx, weight in enumerate(weights.get(agent_name, [])):
                writer.writerow(
                    {
                        "agent": agent_name,
                        "feature_index": feat_idx,
                        "weight": weight,
                    }
                )

    stake_fields = ["cluster_id"] + [f"stake_{name}" for name in agent_names]
    stakes_path = policy_dir / "stakes.csv"
    with stakes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stake_fields)
        writer.writeheader()
        for cluster_id in sorted(stakes):
            row: Dict[str, Any] = {"cluster_id": cluster_id}
            for agent_name in agent_names:
                row[f"stake_{agent_name}"] = stakes[cluster_id].get(agent_name, 0.0)
            writer.writerow(row)

    executors_path = policy_dir / "executors.csv"
    with executors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed_id", "cluster_id", "executor_agent"]
        )
        writer.writeheader()
        for seed in seeds:
            writer.writerow(
                {
                    "seed_id": seed.seed_id,
                    "cluster_id": seed.cluster_id,
                    "executor_agent": executors.get(seed.cluster_id, ""),
                }
            )


def write_policy_outputs(
    policy_name: str,
    seeds: List[SeedResult],
    cluster_stats: ClusterStats,
    results_dir: Path,
    include_cluster_metadata: bool = True,
) -> Path:
    """Write all outputs for a single policy into its subdirectory.

    Parameters
    ----------
    policy_name : str
        Policy identifier used as subdirectory name.
    seeds : list of SeedResult
        Seeds selected by the policy.
    cluster_stats : dict
        Global cluster statistics (same for all policies). Ignored when
        ``include_cluster_metadata`` is ``False``.
    results_dir : Path
        Top-level results directory.
    include_cluster_metadata : bool
        Whether to write per-policy cluster ``metadata.csv``.

    Returns
    -------
    Path
        Policy output directory path.
    """
    policy_dir = ensure_dir(results_dir / policy_name)
    write_seeds_csv(seeds, policy_dir)
    if include_cluster_metadata and cluster_stats:
        write_cluster_statistics(cluster_stats, policy_dir)
    return policy_dir
