"""Configuration schema and validation for AdaptivePy runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_CLUSTERING_METHOD = "kmeans"
DEFAULT_SEED_SELECTION = "nearest_center"
DEFAULT_N_SEEDS = 10
DEFAULT_RANDOM_SEED = 42


@dataclass
class ClusteringConfig:
    """Clustering backend configuration.

    Attributes
    ----------
    method : str
        Clustering method name: ``kmeans``, ``minibatch_kmeans``, or
        ``regular_space``.
    n_clusters : int
        Number of clusters to fit.
    params : dict
        Additional keyword arguments passed to the clusterer.
    """

    method: str = DEFAULT_CLUSTERING_METHOD
    n_clusters: int = 10
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedSelectionConfig:
    """Frame-level seed selection configuration.

    Attributes
    ----------
    method : str
        Selection method: ``nearest_center`` or ``random_frame``.
    """

    method: str = DEFAULT_SEED_SELECTION


@dataclass
class RunConfig:
    """Full configuration for an adaptive sampling run.

    Attributes
    ----------
    features_dir : Path
        Directory containing ``*.npy`` feature files.
    output_dir : Path
        Directory where results are written.
    trajectories_dir : Path or None
        Optional directory containing coordinate trajectories.
    topology : Path or None
        Topology file required when trajectories are provided.
    clustering : ClusteringConfig
        Clustering settings.
    policies : list of str
        Policy names to evaluate in parallel.
    n_seeds : int
        Number of seed frames to select per policy.
    seed_selection : SeedSelectionConfig
        Frame selection method within chosen clusters.
    random_seed : int
        Global random seed for reproducibility.
    write_pdbs : bool
        Whether to write PDB files when trajectories are available.
    """

    features_dir: Path
    output_dir: Path
    trajectories_dir: Optional[Path] = None
    topology: Optional[Path] = None
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    policies: List[str] = field(default_factory=lambda: ["least_counts"])
    n_seeds: int = DEFAULT_N_SEEDS
    seed_selection: SeedSelectionConfig = field(default_factory=SeedSelectionConfig)
    random_seed: int = DEFAULT_RANDOM_SEED
    write_pdbs: bool = True


def load_config(path: str | Path) -> RunConfig:
    """Load and parse a YAML run configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    RunConfig
        Parsed and validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "features_dir" not in raw:
        raise ValueError("Configuration must specify 'features_dir'.")
    if "output_dir" not in raw:
        raise ValueError("Configuration must specify 'output_dir'.")

    features_dir = Path(raw["features_dir"])
    output_dir = Path(raw["output_dir"])
    trajectories_dir = (
        Path(raw["trajectories_dir"]) if raw.get("trajectories_dir") else None
    )
    topology = Path(raw["topology"]) if raw.get("topology") else None

    if trajectories_dir is not None and topology is None:
        raise ValueError(
            "When 'trajectories_dir' is provided, 'topology' must also be set."
        )

    clustering_raw = raw.get("clustering", {})
    clustering = ClusteringConfig(
        method=clustering_raw.get("method", DEFAULT_CLUSTERING_METHOD),
        n_clusters=int(clustering_raw.get("n_clusters", 10)),
        params=dict(clustering_raw.get("params", {})),
    )

    seed_raw = raw.get("seed_selection", {})
    seed_selection = SeedSelectionConfig(
        method=seed_raw.get("method", DEFAULT_SEED_SELECTION),
    )

    policies = raw.get("policies", ["least_counts"])
    if isinstance(policies, str):
        policies = [policies]

    return RunConfig(
        features_dir=features_dir,
        output_dir=output_dir,
        trajectories_dir=trajectories_dir,
        topology=topology,
        clustering=clustering,
        policies=policies,
        n_seeds=int(raw.get("n_seeds", DEFAULT_N_SEEDS)),
        seed_selection=seed_selection,
        random_seed=int(raw.get("random_seed", DEFAULT_RANDOM_SEED)),
        write_pdbs=bool(raw.get("write_pdbs", True)),
    )


def config_to_dict(config: RunConfig) -> Dict[str, Any]:
    """Convert a :class:`RunConfig` to a plain dictionary for serialization.

    Parameters
    ----------
    config : RunConfig
        Configuration object to serialize.

    Returns
    -------
    dict
        YAML-serializable configuration dictionary.
    """
    result: Dict[str, Any] = {
        "features_dir": str(config.features_dir),
        "output_dir": str(config.output_dir),
        "clustering": {
            "method": config.clustering.method,
            "n_clusters": config.clustering.n_clusters,
            "params": config.clustering.params,
        },
        "policies": list(config.policies),
        "n_seeds": config.n_seeds,
        "seed_selection": {"method": config.seed_selection.method},
        "random_seed": config.random_seed,
        "write_pdbs": config.write_pdbs,
    }
    if config.trajectories_dir is not None:
        result["trajectories_dir"] = str(config.trajectories_dir)
    if config.topology is not None:
        result["topology"] = str(config.topology)
    return result
