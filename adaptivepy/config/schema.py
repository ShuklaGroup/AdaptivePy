"""Configuration schema and validation for AdaptivePy runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
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
class MetapolicyConfig:
    """Optional ensemble policy configuration."""

    enabled: bool = False
    name: str = "metapolicy"
    strategy: str = "majority_polling"
    policies: List[str] = field(default_factory=list)
    n_seeds: Optional[int] = None
    rank_depth: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=dict)
    allocations: Dict[str, int] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Full configuration for an adaptive sampling run.

    Attributes
    ----------
    features_dir : Path
        Directory containing ``*.npy`` or ``*.pkl`` feature files.
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
    policy_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metapolicy: MetapolicyConfig = field(default_factory=MetapolicyConfig)


def _parse_policy_params(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Parse optional per-policy parameter blocks from YAML."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'policy_params' must be a mapping of policy names to settings.")
    return {str(name): dict(params or {}) for name, params in raw.items()}


def _parse_policy_list(raw: Any, field_name: str) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"'{field_name}' must be a non-empty policy list.")
    return [str(policy) for policy in raw]


def _parse_metapolicy(
    raw: Any,
    default_policies: List[str],
    default_n_seeds: int,
) -> MetapolicyConfig:
    """Parse and validate the optional metapolicy block."""
    if raw is None:
        return MetapolicyConfig()
    if not isinstance(raw, dict):
        raise ValueError("'metapolicy' must be a mapping.")

    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return MetapolicyConfig(enabled=False)

    strategy = str(raw.get("strategy", "majority_polling"))
    if strategy not in {"majority_polling", "allocation"}:
        raise ValueError(
            "metapolicy.strategy must be 'majority_polling' or 'allocation'."
        )

    policies = _parse_policy_list(
        raw.get("policies", default_policies), "metapolicy.policies"
    )
    n_seeds = int(raw.get("n_seeds", default_n_seeds))
    if n_seeds < 1:
        raise ValueError("metapolicy.n_seeds must be >= 1.")

    rank_depth = raw.get("rank_depth")
    if rank_depth is not None:
        rank_depth = int(rank_depth)
        if rank_depth < 1:
            raise ValueError("metapolicy.rank_depth must be >= 1.")

    weights_raw = raw.get("weights", {})
    if weights_raw is None:
        weights_raw = {}
    if not isinstance(weights_raw, dict):
        raise ValueError("metapolicy.weights must map policy names to weights.")
    weights = {str(name): float(value) for name, value in weights_raw.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("metapolicy.weights must be non-negative.")

    allocations: Dict[str, int] = {}
    if strategy == "allocation":
        allocations_raw = raw.get("allocations")
        if not isinstance(allocations_raw, dict):
            raise ValueError(
                "metapolicy.allocations must map every policy to a positive integer."
            )
        missing = set(policies) - set(str(name) for name in allocations_raw)
        extra = set(str(name) for name in allocations_raw) - set(policies)
        if missing:
            raise ValueError(
                f"metapolicy.allocations missing policies: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"metapolicy.allocations contains unknown policies: {sorted(extra)}"
            )
        allocations = {
            str(name): int(value) for name, value in allocations_raw.items()
        }
        if any(value < 1 for value in allocations.values()):
            raise ValueError("metapolicy.allocations must be positive integers.")
        if sum(allocations.values()) != n_seeds:
            raise ValueError(
                "metapolicy allocation sum must equal metapolicy.n_seeds."
            )

    return MetapolicyConfig(
        enabled=True,
        name=str(raw.get("name", "metapolicy")),
        strategy=strategy,
        policies=policies,
        n_seeds=n_seeds,
        rank_depth=rank_depth,
        weights=weights,
        allocations=allocations,
    )


def validate_fast_policy_params(
    params: Dict[str, Any],
    n_features: int,
) -> Dict[str, Any]:
    """Validate and normalize FAST policy parameters.

    Parameters
    ----------
    params : dict
        Raw FAST policy settings from configuration.
    n_features : int
        Number of feature dimensions in the loaded dataset.

    Returns
    -------
    dict
        Normalized keyword arguments for :class:`FastPolicy`.

    Raises
    ------
    ValueError
        If required FAST settings are missing or invalid.
    """
    if "feature_indices" not in params:
        raise ValueError(
            "FAST policy requires 'feature_indices' in policy_params.fast."
        )

    feature_indices = params["feature_indices"]
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]
    if not isinstance(feature_indices, (list, tuple)) or not feature_indices:
        raise ValueError("FAST 'feature_indices' must be a non-empty list of integers.")

    normalized_indices: List[int] = []
    for index in feature_indices:
        if not isinstance(index, int):
            raise ValueError("FAST 'feature_indices' must contain integers only.")
        if index < 0 or index >= n_features:
            raise ValueError(
                f"FAST feature index {index} is out of range for "
                f"{n_features} feature dimensions."
            )
        normalized_indices.append(index)

    directions = params.get("directions")
    if directions is not None:
        if isinstance(directions, str):
            directions = [directions]
        if not isinstance(directions, (list, tuple)):
            raise ValueError("FAST 'directions' must be a list of strings.")
        if len(directions) != len(normalized_indices):
            raise ValueError("FAST 'directions' must match 'feature_indices' length.")
        for direction in directions:
            if direction not in {"maximize", "minimize"}:
                raise ValueError(
                    f"Invalid FAST direction '{direction}'. "
                    "Use 'maximize' or 'minimize'."
                )

    weights = params.get("weights")
    if weights is not None:
        if not isinstance(weights, (list, tuple)):
            raise ValueError("FAST 'weights' must be a list of numbers.")
        if len(weights) != len(normalized_indices):
            raise ValueError("FAST 'weights' must match 'feature_indices' length.")
        weights = [float(w) for w in weights]
        if any(w < 0 for w in weights):
            raise ValueError("FAST 'weights' must be non-negative.")
        if sum(weights) <= 0:
            raise ValueError("FAST 'weights' must sum to a positive value.")

    alpha = float(params.get("alpha", 1.0))
    if alpha < 0:
        raise ValueError("FAST 'alpha' must be non-negative.")

    kwargs: Dict[str, Any] = {
        "feature_indices": normalized_indices,
        "alpha": alpha,
    }
    if directions is not None:
        kwargs["directions"] = list(directions)
    if weights is not None:
        kwargs["weights"] = weights
    return kwargs


def validate_ma_reap_policy_params(
    params: Dict[str, Any],
    traj_names: Sequence[str],
    n_features: int,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and normalize MA-REAP policy parameters.

    Parameters
    ----------
    params : dict
        Raw MA-REAP settings from configuration.
    traj_names : sequence of str
        Feature file stems in the loaded dataset.
    n_features : int
        Number of feature dimensions.
    n_seeds : int
        Seeds to select per policy run.
    n_clusters : int or None
        Number of populated clusters after clustering.

    Returns
    -------
    dict
        Normalized keyword arguments for :class:`MaReapPolicy`.
    """
    if "agents" not in params:
        raise ValueError(
            "MA-REAP policy requires 'agents' in policy_params.ma_reap."
        )

    agents_raw = params["agents"]
    if not isinstance(agents_raw, dict) or len(agents_raw) < 2:
        raise ValueError(
            "MA-REAP 'agents' must map at least two agent names to trajectory lists."
        )

    agent_assignments: Dict[str, List[str]] = {}
    seen_stems: set[str] = set()
    for agent_name, stems in agents_raw.items():
        if isinstance(stems, str):
            stems = [stems]
        if not isinstance(stems, (list, tuple)) or not stems:
            raise ValueError(
                f"MA-REAP agent '{agent_name}' must have a non-empty trajectory list."
            )
        normalized_stems = [str(s) for s in stems]
        for stem in normalized_stems:
            if stem not in traj_names:
                raise ValueError(
                    f"MA-REAP agent '{agent_name}' references unknown trajectory '{stem}'."
                )
            if stem in seen_stems:
                raise ValueError(
                    f"Trajectory '{stem}' is assigned to multiple MA-REAP agents."
                )
            seen_stems.add(stem)
        agent_assignments[str(agent_name)] = normalized_stems

    unassigned = set(traj_names) - seen_stems
    if unassigned:
        raise ValueError(
            "MA-REAP requires every trajectory to be assigned to an agent. "
            f"Unassigned: {sorted(unassigned)}"
        )

    n_candidates = params.get("n_candidates")
    if n_candidates is None:
        n_candidates = max(n_seeds, n_seeds * 3)
    n_candidates = int(n_candidates)
    if n_candidates < n_seeds:
        raise ValueError(
            f"MA-REAP n_candidates ({n_candidates}) must be >= n_seeds ({n_seeds})."
        )
    if n_clusters is not None:
        if n_clusters < n_seeds:
            raise ValueError(
                f"MA-REAP requires at least {n_seeds} clusters, got {n_clusters}."
            )
        if n_candidates > n_clusters:
            n_candidates = n_clusters

    delta = float(params.get("delta", 0.05))
    if not (0.0 < delta < 1.0):
        raise ValueError("MA-REAP 'delta' must be in (0, 1).")

    stakes_method = str(params.get("stakes_method", "percentage"))
    if stakes_method not in {"percentage", "equal", "max", "logistic"}:
        raise ValueError(
            "MA-REAP 'stakes_method' must be one of: "
            "percentage, equal, max, logistic."
        )

    stakes_k = params.get("stakes_k")
    if stakes_method == "logistic":
        if stakes_k is None:
            raise ValueError(
                "MA-REAP 'stakes_k' is required when stakes_method is 'logistic'."
            )
        stakes_k = float(stakes_k)
    elif stakes_k is not None:
        stakes_k = float(stakes_k)

    regime = str(params.get("regime", "collaborative"))
    if regime not in {"collaborative", "noncollaborative", "competitive"}:
        raise ValueError(
            "MA-REAP 'regime' must be collaborative, noncollaborative, or competitive."
        )

    initial_weights = params.get("initial_weights")
    if initial_weights is not None:
        initial_weights = np.asarray(initial_weights, dtype=float)
        n_agents = len(agent_assignments)
        valid_1d = initial_weights.ndim == 1 and initial_weights.shape[0] == n_features
        valid_2d = (
            initial_weights.ndim == 2
            and initial_weights.shape == (n_agents, n_features)
        )
        if not (valid_1d or valid_2d):
            raise ValueError(
                f"MA-REAP initial_weights must have shape ({n_features},) or "
                f"({n_agents}, {n_features})."
            )
        if np.any(initial_weights < 0):
            raise ValueError("MA-REAP initial_weights must be non-negative.")
        if valid_1d and not np.isclose(initial_weights.sum(), 1.0):
            raise ValueError("MA-REAP initial_weights must sum to 1.")
        if valid_2d and not np.allclose(initial_weights.sum(axis=1), 1.0):
            raise ValueError("MA-REAP initial_weights rows must each sum to 1.")

    kwargs: Dict[str, Any] = {
        "agent_assignments": agent_assignments,
        "traj_names": list(traj_names),
        "n_candidates": n_candidates,
        "delta": delta,
        "stakes_method": stakes_method,
        "regime": regime,
    }
    if stakes_k is not None:
        kwargs["stakes_k"] = stakes_k
    if initial_weights is not None:
        kwargs["initial_weights"] = initial_weights
    return kwargs


def validate_knn_as_policy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize kNN-AS policy parameters.

    Parameters
    ----------
    params : dict
        Raw kNN-AS policy settings from configuration.

    Returns
    -------
    dict
        Normalized keyword arguments for :class:`KnnAsPolicy`.
    """
    k = int(params.get("k", 5))
    if k < 2:
        raise ValueError("kNN-AS 'k' must be >= 2.")

    scoring = str(params.get("scoring", "vectorsum"))
    if scoring not in {"vectorsum", "distance"}:
        raise ValueError("kNN-AS 'scoring' must be 'vectorsum' or 'distance'.")

    return {"k": k, "scoring": scoring}


def validate_maxent_vampnet_policy_params(
    params: Dict[str, Any],
    n_features: int,
    traj_index_map: Optional[Dict[int, tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Validate and normalize MaxEnt VAMPNet policy parameters.

    Parameters
    ----------
    params : dict
        Raw MaxEnt VAMPNet settings from configuration.
    n_features : int
        Number of feature dimensions in the loaded dataset.
    traj_index_map : dict or None
        Mapping from trajectory ID to ``(start, end)`` indices in the
        concatenated feature matrix.

    Returns
    -------
    dict
        Normalized keyword arguments for :class:`MaxEntVampNetPolicy`.
    """
    if "n_states" in params and "output_states" in params:
        if int(params["n_states"]) != int(params["output_states"]):
            raise ValueError(
                "MaxEnt VAMPNet 'n_states' and 'output_states' must match "
                "when both are provided."
            )
    n_states_raw = params.get("n_states", params.get("output_states", n_features))
    n_states = int(n_states_raw)
    if n_states < 2:
        raise ValueError("MaxEnt VAMPNet 'n_states' must be >= 2.")

    lagtime = int(params.get("lagtime", 1))
    if lagtime < 1:
        raise ValueError("MaxEnt VAMPNet 'lagtime' must be >= 1.")

    if traj_index_map:
        for traj_id, (start, end) in sorted(traj_index_map.items()):
            n_frames = end - start
            if n_frames <= lagtime:
                raise ValueError(
                    f"Trajectory {traj_id} has {n_frames} frames, but lagtime "
                    f"requires at least {lagtime + 1} frames."
                )

    hidden_layers = params.get("hidden_layers")
    if hidden_layers is not None:
        if not isinstance(hidden_layers, (list, tuple)) or not hidden_layers:
            raise ValueError(
                "MaxEnt VAMPNet 'hidden_layers' must be a non-empty list of integers."
            )
        normalized_layers = [int(width) for width in hidden_layers]
        if any(width < 1 for width in normalized_layers):
            raise ValueError(
                "MaxEnt VAMPNet 'hidden_layers' must contain positive integers."
            )
    else:
        normalized_layers = None

    learning_rate = float(params.get("learning_rate", 1e-4))
    if learning_rate <= 0:
        raise ValueError("MaxEnt VAMPNet 'learning_rate' must be positive.")

    batch_size = int(params.get("batch_size", 2048))
    if batch_size < 1:
        raise ValueError("MaxEnt VAMPNet 'batch_size' must be >= 1.")

    epochs = int(params.get("epochs", 100))
    if epochs < 1:
        raise ValueError("MaxEnt VAMPNet 'epochs' must be >= 1.")

    device = str(params.get("device", "cpu"))
    num_threads = int(params.get("num_threads", 1))
    if num_threads < 1:
        raise ValueError("MaxEnt VAMPNet 'num_threads' must be >= 1.")

    kwargs: Dict[str, Any] = {
        "n_features": n_features,
        "n_states": n_states,
        "lagtime": lagtime,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": device,
        "num_threads": num_threads,
    }
    if normalized_layers is not None:
        kwargs["hidden_layers"] = normalized_layers
    return kwargs


def validate_ts_dar_policy_params(
    params: Dict[str, Any],
    n_features: int,
    traj_index_map: Optional[Dict[int, tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Validate and normalize TS-DAR policy parameters."""
    n_states = int(params.get("n_states", min(max(2, n_features), 4)))
    if n_states < 2:
        raise ValueError("TS-DAR 'n_states' must be >= 2.")

    latent_dim = int(params.get("latent_dim", 2 if n_states <= 3 else 3))
    if latent_dim < 1:
        raise ValueError("TS-DAR 'latent_dim' must be >= 1.")

    hidden_layers = params.get("hidden_layers")
    encoder_sizes = params.get("encoder_sizes")
    if hidden_layers is not None:
        if not isinstance(hidden_layers, (list, tuple)):
            raise ValueError("TS-DAR 'hidden_layers' must be a list of integers.")
        normalized_layers = [int(width) for width in hidden_layers]
    elif encoder_sizes is not None:
        if not isinstance(encoder_sizes, (list, tuple)) or len(encoder_sizes) < 2:
            raise ValueError(
                "TS-DAR 'encoder_sizes' must contain at least input and latent sizes."
            )
        normalized_encoder = [int(size) for size in encoder_sizes]
        if any(size < 1 for size in normalized_encoder):
            raise ValueError("TS-DAR 'encoder_sizes' must contain positive integers.")
        if normalized_encoder[0] != n_features:
            raise ValueError("TS-DAR 'encoder_sizes' first value must equal n_features.")
        if normalized_encoder[-1] != latent_dim:
            raise ValueError("TS-DAR 'encoder_sizes' last value must equal latent_dim.")
        normalized_layers = normalized_encoder[1:-1]
    else:
        normalized_layers = [128, 64]
    if not isinstance(normalized_layers, list) or any(
        width < 1 for width in normalized_layers
    ):
        raise ValueError("TS-DAR 'hidden_layers' must contain positive integers.")

    if hidden_layers is not None and encoder_sizes is not None:
        expected_encoder = [n_features, *normalized_layers, latent_dim]
        normalized_encoder = [int(size) for size in encoder_sizes]
        if normalized_encoder != expected_encoder:
            raise ValueError(
                "TS-DAR 'hidden_layers' and 'encoder_sizes' describe "
                "different architectures."
            )

    lagtime = int(params.get("lagtime", 1))
    if lagtime < 1:
        raise ValueError("TS-DAR 'lagtime' must be >= 1.")

    if traj_index_map:
        total_pairs = 0
        for traj_id, (start, end) in sorted(traj_index_map.items()):
            n_frames = end - start
            if n_frames <= lagtime:
                raise ValueError(
                    f"Trajectory {traj_id} has {n_frames} frames, but lagtime "
                    f"requires at least {lagtime + 1} frames."
                )
            total_pairs += n_frames - lagtime
        if total_pairs < 2:
            raise ValueError("TS-DAR requires at least two lagged frame pairs.")

    learning_rate = float(params.get("learning_rate", 1e-3))
    if learning_rate <= 0:
        raise ValueError("TS-DAR 'learning_rate' must be positive.")

    batch_size = int(params.get("batch_size", 2048))
    if batch_size < 1:
        raise ValueError("TS-DAR 'batch_size' must be >= 1.")

    epochs = int(params.get("epochs", 100))
    if epochs < 1:
        raise ValueError("TS-DAR 'epochs' must be >= 1.")

    pretrain = int(params.get("pretrain", 10))
    if pretrain < 0:
        raise ValueError("TS-DAR 'pretrain' must be >= 0.")

    beta = float(params.get("beta", 0.01))
    if beta < 0:
        raise ValueError("TS-DAR 'beta' must be non-negative.")

    gamma = float(params.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("TS-DAR 'gamma' must be positive.")

    scaling_temperature = float(params.get("scaling_temperature", 0.1))
    if scaling_temperature <= 0:
        raise ValueError("TS-DAR 'scaling_temperature' must be positive.")

    proto_update_factor = float(params.get("proto_update_factor", 0.5))
    if not (0.0 <= proto_update_factor <= 1.0):
        raise ValueError("TS-DAR 'proto_update_factor' must be in [0, 1].")

    optimizer = str(params.get("optimizer", "Adam"))
    if optimizer not in {"Adam", "SGD", "RMSprop"}:
        raise ValueError("TS-DAR 'optimizer' must be one of: Adam, SGD, RMSprop.")

    device = str(params.get("device", "cpu"))
    num_threads = int(params.get("num_threads", 1))
    if num_threads < 1:
        raise ValueError("TS-DAR 'num_threads' must be >= 1.")

    train_split = float(params.get("train_split", 0.9))
    if not (0.0 < train_split <= 1.0):
        raise ValueError("TS-DAR 'train_split' must be in (0, 1].")

    return {
        "n_features": n_features,
        "n_states": n_states,
        "latent_dim": latent_dim,
        "hidden_layers": normalized_layers,
        "lagtime": lagtime,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "pretrain": pretrain,
        "beta": beta,
        "gamma": gamma,
        "scaling_temperature": scaling_temperature,
        "proto_update_factor": proto_update_factor,
        "optimizer": optimizer,
        "device": device,
        "num_threads": num_threads,
        "train_split": train_split,
    }


def build_policy_kwargs(
    policy_name: str,
    config: RunConfig,
    n_features: Optional[int] = None,
    traj_names: Optional[Sequence[str]] = None,
    n_clusters: Optional[int] = None,
    n_seeds: Optional[int] = None,
    traj_index_map: Optional[Dict[int, tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Build constructor keyword arguments for a configured policy.

    Parameters
    ----------
    policy_name : str
        Registered policy name.
    config : RunConfig
        Parsed run configuration.
    n_features : int or None
        Feature dimensionality for policies that require it.

    Returns
    -------
    dict
        Keyword arguments forwarded to :func:`get_policy`.
    """
    kwargs: Dict[str, Any] = {}
    params = config.policy_params.get(policy_name, {})
    seeds = n_seeds if n_seeds is not None else config.n_seeds

    if policy_name == "random":
        kwargs["random_state"] = config.random_seed
    elif policy_name == "fast":
        if n_features is None:
            raise ValueError("FAST policy validation requires feature dimensionality.")
        kwargs.update(validate_fast_policy_params(params, n_features))
    elif policy_name == "ma_reap":
        if n_features is None or traj_names is None:
            raise ValueError(
                "MA-REAP policy validation requires feature dimensionality and traj_names."
            )
        kwargs.update(
            validate_ma_reap_policy_params(
                params,
                traj_names=traj_names,
                n_features=n_features,
                n_seeds=seeds,
                n_clusters=n_clusters,
            )
        )
    elif policy_name == "knn_as":
        kwargs.update(validate_knn_as_policy_params(params))
    elif policy_name == "maxent_vampnet":
        if n_features is None:
            raise ValueError(
                "MaxEnt VAMPNet policy validation requires feature dimensionality."
            )
        kwargs.update(
            validate_maxent_vampnet_policy_params(
                params,
                n_features=n_features,
                traj_index_map=traj_index_map,
            )
        )
    elif policy_name == "ts_dar":
        if n_features is None:
            raise ValueError("TS-DAR policy validation requires feature dimensionality.")
        kwargs.update(
            validate_ts_dar_policy_params(
                params,
                n_features=n_features,
                traj_index_map=traj_index_map,
            )
        )
        kwargs["random_state"] = config.random_seed

    return kwargs


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

    policies = _parse_policy_list(raw.get("policies", ["least_counts"]), "policies")

    policy_params = _parse_policy_params(raw.get("policy_params"))
    n_seeds = int(raw.get("n_seeds", DEFAULT_N_SEEDS))
    metapolicy = _parse_metapolicy(raw.get("metapolicy"), policies, n_seeds)

    return RunConfig(
        features_dir=features_dir,
        output_dir=output_dir,
        trajectories_dir=trajectories_dir,
        topology=topology,
        clustering=clustering,
        policies=policies,
        n_seeds=n_seeds,
        seed_selection=seed_selection,
        random_seed=int(raw.get("random_seed", DEFAULT_RANDOM_SEED)),
        write_pdbs=bool(raw.get("write_pdbs", True)),
        policy_params=policy_params,
        metapolicy=metapolicy,
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
    if config.policy_params:
        result["policy_params"] = config.policy_params
    if config.metapolicy.enabled:
        metapolicy: Dict[str, Any] = {
            "enabled": True,
            "name": config.metapolicy.name,
            "strategy": config.metapolicy.strategy,
            "policies": list(config.metapolicy.policies),
            "n_seeds": config.metapolicy.n_seeds,
        }
        if config.metapolicy.rank_depth is not None:
            metapolicy["rank_depth"] = config.metapolicy.rank_depth
        if config.metapolicy.weights:
            metapolicy["weights"] = config.metapolicy.weights
        if config.metapolicy.allocations:
            metapolicy["allocations"] = config.metapolicy.allocations
        result["metapolicy"] = metapolicy
    return result
