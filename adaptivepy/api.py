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
from adaptivepy.models import SeedResult
from adaptivepy.output.pdb_writer import write_seed_pdbs
from adaptivepy.output.writer import (
    write_assignments,
    write_cluster_model,
    write_cluster_statistics,
    write_combined_metadata,
    write_fast_scores,
    write_ma_reap_outputs,
    write_policy_outputs,
    write_run_config,
)
from adaptivepy.policies import get_policy
from adaptivepy.selection.frame_selector import select_seeds
from adaptivepy.stats.cluster_stats import assign_clusters, compute_cluster_stats
from adaptivepy.utils.io_utils import ensure_dir
from adaptivepy.utils.logging import setup_logger

logger = logging.getLogger(__name__)


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

    # --- Clustering ---
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
    n_features = dataset.feature_matrix.shape[1]

    logger.info(
        "Clustering complete: %d clusters, %d total frames",
        len(cluster_stats),
        len(dataset.frames),
    )

    # --- Global artifacts ---
    write_run_config(config, output_dir, config_path)
    write_assignments(labels, output_dir)
    write_cluster_model(clusterer.model, output_dir)
    write_cluster_statistics(cluster_stats, output_dir)

    # --- Policies ---
    policy_seeds: Dict[str, List[SeedResult]] = {}

    for policy_name in config.policies:
        logger.info("Applying policy: %s", policy_name)
        policy_kwargs = build_policy_kwargs(
            policy_name,
            config,
            n_features=n_features,
            traj_names=dataset.traj_names,
            n_clusters=len(cluster_stats),
        )
        if policy_name == "ma_reap":
            policy_kwargs["cluster_centers"] = centers

        policy = get_policy(policy_name, **policy_kwargs)
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
        policy_seeds[policy_name] = seeds

        policy_dir = write_policy_outputs(
            policy_name, seeds, cluster_stats, output_dir
        )

        if policy_name == "fast" and hasattr(policy, "last_scores"):
            write_fast_scores(policy.last_scores, policy_dir)

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

        if (
            config.write_pdbs
            and trajectory_map is not None
            and config.topology is not None
        ):
            write_seed_pdbs(
                seeds, config.topology, trajectory_map, policy_dir
            )

        logger.info("Policy %s selected %d seeds", policy_name, len(seeds))

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

    for policy_name in config.policies:
        get_policy(
            policy_name,
            **build_policy_kwargs(
                policy_name,
                config,
                n_features=n_features,
                traj_names=dataset.traj_names,
            ),
        )

    return config
