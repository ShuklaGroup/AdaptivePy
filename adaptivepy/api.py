"""High-level API orchestrating the adaptive sampling workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from adaptivepy.clustering import create_clusterer, fit_clusterer
from adaptivepy.config.schema import RunConfig, build_policy_kwargs, load_config
from adaptivepy.io.loader import (
    list_feature_files,
    list_trajectory_files,
    load_features,
    validate_dataset,
    validate_feature_trajectory_mapping,
)
from adaptivepy.io.trajectory import (
    build_trajectory_map,
    validate_trajectory_frame_counts,
)
from adaptivepy.metapolicies import (
    MetapolicyDecision,
    allocation,
    cluster_policy_ranking,
    majority_polling,
    maxent_cluster_ranking,
    ts_dar_cluster_ranking,
)
from adaptivepy.models import SeedResult
from adaptivepy.output.pdb_writer import write_seed_pdbs
from adaptivepy.output.writer import (
    write_assignments,
    write_cluster_model,
    write_cluster_statistics,
    write_combined_metadata,
    write_fast_scores,
    write_knn_as_scores,
    write_ma_reap_outputs,
    write_metapolicy_votes,
    write_maxent_vampnet_scores,
    write_policy_outputs,
    write_run_config,
    write_ts_dar_scores,
)
from adaptivepy.policies import get_policy
from adaptivepy.policies.base import policy_requires_clustering
from adaptivepy.selection.frame_selector import select_seeds
from adaptivepy.stats.cluster_stats import assign_clusters, compute_cluster_stats
from adaptivepy.utils.io_utils import ensure_dir
from adaptivepy.utils.logging import setup_logger

logger = logging.getLogger(__name__)


def _unique_policies(policies: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for policy_name in policies:
        if policy_name in seen:
            continue
        seen.add(policy_name)
        result.append(policy_name)
    return result


def run_adaptive_sampling(
    config_path: str | Path,
    config: Optional[RunConfig] = None,
) -> Dict[str, List[SeedResult]]:
    """Execute a full adaptive sampling run from a YAML configuration.

    Workflow
    --------
    1. Load features (and optionally validate trajectories).
    2. Cluster the concatenated feature matrix.
    3. Compute cluster statistics.
    4. Apply each configured policy and select seed frames.
    5. Write metadata, assignments, model, and optional PDBs.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    config : RunConfig or None
        Pre-parsed configuration. If ``None``, loaded from ``config_path``.

    Returns
    -------
    dict
        Mapping from policy name to lists of :class:`SeedResult`.

    Raises
    ------
    FileNotFoundError
        If required input paths do not exist.
    ValueError
        If validation checks fail.
    """
    config_path = Path(config_path)
    if config is None:
        config = load_config(config_path)

    output_dir = ensure_dir(config.output_dir)
    log_path = output_dir / "logs.txt"
    setup_logger("adaptivepy", log_file=log_path)
    logger.info("Starting AdaptivePy run with config %s", config_path)

    np.random.seed(config.random_seed)

    # --- Load and validate data ---
    feature_files = list_feature_files(config.features_dir)
    trajectory_files: Optional[List[Path]] = None
    trajectory_map: Optional[Dict[int, Path]] = None

    if config.trajectories_dir is not None:
        trajectory_files = list_trajectory_files(config.trajectories_dir)
        validate_feature_trajectory_mapping(feature_files, trajectory_files)

    dataset = load_features(config.features_dir)
    validate_dataset(dataset, trajectory_files)

    if config.trajectories_dir is not None and config.topology is not None:
        trajectory_map = build_trajectory_map(
            config.trajectories_dir, dataset.traj_names
        )
        expected_counts = {
            traj_id: end - start
            for traj_id, (start, end) in dataset.traj_index_map.items()
        }
        validate_trajectory_frame_counts(
            config.topology, trajectory_map, expected_counts
        )

    execution_policies = list(config.policies)
    if config.metapolicy.enabled:
        execution_policies.extend(config.metapolicy.policies)
        if config.metapolicy.name in execution_policies:
            raise ValueError(
                f"metapolicy.name '{config.metapolicy.name}' conflicts with a "
                "base policy name."
            )
    execution_policies = _unique_policies(execution_policies)

    # --- Clustering (optional for frame-level policies) ---
    needs_clustering = any(
        policy_requires_clustering(policy_name) for policy_name in execution_policies
    ) or config.metapolicy.enabled
    cluster_stats = {}
    centers = None
    labels = None
    n_features = dataset.feature_matrix.shape[1]

    if needs_clustering:
        clusterer = create_clusterer(
            method=config.clustering.method,
            n_clusters=config.clustering.n_clusters,
            random_state=config.random_seed,
            params=config.clustering.params,
        )
        fit_clusterer(clusterer, dataset.feature_matrix)
        labels = clusterer.predict(dataset.feature_matrix)
        assign_clusters(dataset, labels)

        cluster_stats = compute_cluster_stats(dataset)
        centers = clusterer.cluster_centers_

        logger.info(
            "Clustering complete: %d clusters, %d total frames",
            len(cluster_stats),
            len(dataset.frames),
        )
    else:
        logger.info(
            "Skipping clustering for frame-level policies (%d total frames)",
            len(dataset.frames),
        )

    # --- Global artifacts ---
    write_run_config(config, output_dir, config_path)
    if needs_clustering and labels is not None:
        write_assignments(labels, output_dir)
        write_cluster_model(clusterer.model, output_dir)
        write_cluster_statistics(cluster_stats, output_dir)

    # --- Policies ---
    policy_seeds: Dict[str, List[SeedResult]] = {}

    policy_objects = {}

    for policy_name in execution_policies:
        logger.info("Applying policy: %s", policy_name)
        policy_kwargs = build_policy_kwargs(
            policy_name,
            config,
            n_features=n_features,
            traj_names=dataset.traj_names,
            n_clusters=len(cluster_stats) if cluster_stats else None,
            traj_index_map=dataset.traj_index_map,
        )
        if policy_name == "ma_reap":
            policy_kwargs["cluster_centers"] = centers
        if policy_name == "knn_as":
            policy_kwargs["cluster_centers"] = centers

        policy = get_policy(policy_name, **policy_kwargs)
        policy_objects[policy_name] = policy
        if policy.requires_clustering:
            selected_clusters = policy.select_clusters(
                cluster_stats, config.n_seeds
            )
            seeds = select_seeds(
                policy_name=policy_name,
                selected_clusters=selected_clusters,
                cluster_stats=cluster_stats,
                cluster_centers=centers,
                method=config.seed_selection.method,
                random_state=config.random_seed,
            )
        else:
            seeds = policy.select_frames(dataset, config.n_seeds)

        policy_seeds[policy_name] = seeds

        policy_dir = write_policy_outputs(
            policy_name,
            seeds,
            cluster_stats,
            output_dir,
            include_cluster_metadata=policy.requires_clustering,
        )

        if policy_name == "fast" and hasattr(policy, "last_scores"):
            write_fast_scores(policy.last_scores, policy_dir)

        if policy_name == "knn_as" and hasattr(policy, "last_scores"):
            write_knn_as_scores(policy.last_scores, policy_dir)

        if policy_name == "ma_reap" and hasattr(policy, "last_scores"):
            write_ma_reap_outputs(
                scores=policy.last_scores,
                weights=policy.last_weights,
                stakes=policy.last_stakes,
                executors=policy.last_executors,
                agent_names=policy.agent_names,
                seeds=seeds,
                output_dir=policy_dir,
            )

        if policy_name == "maxent_vampnet" and hasattr(policy, "last_scores"):
            write_maxent_vampnet_scores(policy.last_scores, policy_dir)

        if policy_name == "ts_dar" and hasattr(policy, "last_scores"):
            write_ts_dar_scores(policy.last_scores, policy_dir)

        if (
            config.write_pdbs
            and trajectory_map is not None
            and config.topology is not None
        ):
            write_seed_pdbs(
                seeds, config.topology, trajectory_map, policy_dir
            )

        logger.info("Policy %s selected %d seeds", policy_name, len(seeds))

    if config.metapolicy.enabled:
        logger.info("Applying metapolicy: %s", config.metapolicy.name)
        if not cluster_stats:
            raise ValueError("Metapolicy requires populated cluster statistics.")

        n_meta_seeds = int(config.metapolicy.n_seeds or config.n_seeds)
        if config.metapolicy.strategy == "allocation":
            default_rank_depth = len(cluster_stats)
        else:
            default_rank_depth = n_meta_seeds
        rank_depth = int(config.metapolicy.rank_depth or default_rank_depth)
        if config.metapolicy.strategy == "allocation":
            rank_depth = max(rank_depth, n_meta_seeds)
        rank_depth = min(len(cluster_stats), rank_depth)
        rankings = []

        for policy_name in config.metapolicy.policies:
            if policy_name == "maxent_vampnet":
                policy = policy_objects[policy_name]
                rankings.append(
                    maxent_cluster_ranking(
                        policy_name=policy_name,
                        scores=getattr(policy, "last_scores", {}),
                        dataset=dataset,
                        cluster_stats=cluster_stats,
                        rank_depth=rank_depth,
                    )
                )
                continue
            if policy_name == "ts_dar":
                policy = policy_objects[policy_name]
                rankings.append(
                    ts_dar_cluster_ranking(
                        policy_name=policy_name,
                        scores=getattr(policy, "last_scores", {}),
                        dataset=dataset,
                        cluster_stats=cluster_stats,
                        rank_depth=rank_depth,
                    )
                )
                continue

            policy_kwargs = build_policy_kwargs(
                policy_name,
                config,
                n_features=n_features,
                traj_names=dataset.traj_names,
                n_clusters=len(cluster_stats),
                traj_index_map=dataset.traj_index_map,
            )
            if policy_name == "ma_reap":
                policy_kwargs["cluster_centers"] = centers
            if policy_name == "knn_as":
                policy_kwargs["cluster_centers"] = centers
            policy = get_policy(policy_name, **policy_kwargs)
            rankings.append(
                cluster_policy_ranking(
                    policy_name=policy_name,
                    policy=policy,
                    cluster_stats=cluster_stats,
                    rank_depth=rank_depth,
                )
            )

        if config.metapolicy.strategy == "allocation":
            decision: MetapolicyDecision = allocation(
                rankings=rankings,
                cluster_stats=cluster_stats,
                allocations=config.metapolicy.allocations,
                n_seeds=n_meta_seeds,
                rank_depth=rank_depth,
                weights=config.metapolicy.weights,
            )
        else:
            decision = majority_polling(
                rankings=rankings,
                cluster_stats=cluster_stats,
                n_seeds=n_meta_seeds,
                rank_depth=rank_depth,
                weights=config.metapolicy.weights,
            )

        seeds = select_seeds(
            policy_name=config.metapolicy.name,
            selected_clusters=decision.selected_clusters,
            cluster_stats=cluster_stats,
            cluster_centers=centers,
            method=config.seed_selection.method,
            random_state=config.random_seed,
        )
        policy_seeds[config.metapolicy.name] = seeds
        policy_dir = write_policy_outputs(
            "metapolicy",
            seeds,
            cluster_stats,
            output_dir,
            include_cluster_metadata=True,
        )
        write_metapolicy_votes(decision.vote_rows, policy_dir)
        if (
            config.write_pdbs
            and trajectory_map is not None
            and config.topology is not None
        ):
            write_seed_pdbs(seeds, config.topology, trajectory_map, policy_dir)

        logger.info(
            "Metapolicy %s selected %d seeds",
            config.metapolicy.name,
            len(seeds),
        )

    if len(policy_seeds) > 1:
        write_combined_metadata(policy_seeds, output_dir)

    logger.info("AdaptivePy run finished. Results written to %s", output_dir)
    return policy_seeds


def validate_config(config_path: str | Path) -> RunConfig:
    """Validate a configuration file and input data without running clustering.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    RunConfig
        Parsed configuration if validation succeeds.

    Raises
    ------
    ValueError
        If validation fails.
    """
    config_path = Path(config_path)
    config = load_config(config_path)

    feature_files = list_feature_files(config.features_dir)

    trajectory_files = None
    if config.trajectories_dir is not None:
        trajectory_files = list_trajectory_files(config.trajectories_dir)
        validate_feature_trajectory_mapping(feature_files, trajectory_files)

    dataset = load_features(config.features_dir)
    validate_dataset(dataset, trajectory_files)
    n_features = dataset.feature_matrix.shape[1]

    if config.trajectories_dir is not None and config.topology is not None:
        trajectory_map = build_trajectory_map(
            config.trajectories_dir, dataset.traj_names
        )
        expected_counts = {
            traj_id: end - start
            for traj_id, (start, end) in dataset.traj_index_map.items()
        }
        validate_trajectory_frame_counts(
            config.topology, trajectory_map, expected_counts
        )

    execution_policies = list(config.policies)
    if config.metapolicy.enabled:
        execution_policies.extend(config.metapolicy.policies)
        if config.metapolicy.name in execution_policies:
            raise ValueError(
                f"metapolicy.name '{config.metapolicy.name}' conflicts with a "
                "base policy name."
            )
    execution_policies = _unique_policies(execution_policies)

    for policy_name in execution_policies:
        get_policy(
            policy_name,
            **build_policy_kwargs(
                policy_name,
                config,
                n_features=n_features,
                traj_names=dataset.traj_names,
                traj_index_map=dataset.traj_index_map,
            ),
        )

    return config
