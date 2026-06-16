"""Integration tests for AdaptivePy."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from adaptivepy.api import run_adaptive_sampling, validate_config
from adaptivepy.config.schema import (
    RunConfig,
    build_policy_kwargs,
    load_config,
    validate_fast_policy_params,
    validate_knn_as_policy_params,
    validate_ma_reap_policy_params,
    validate_maxent_vampnet_policy_params,
)
from adaptivepy.io.loader import load_feature_array, load_features
from adaptivepy.models import Dataset, FrameRecord
from adaptivepy.policies import list_policies
from adaptivepy.policies.fast import FastPolicy, compute_fast_rewards, feature_scale
from adaptivepy.policies.knn_as import KnnAsPolicy, compute_knn_as_scores
from adaptivepy.policies.maxent_vampnet import (
    MaxEntVampNetPolicy,
    compute_shannon_entropy,
    rank_frames_by_entropy,
    split_trajectories_from_dataset,
)
from adaptivepy.policies.ma_reap import (
    MaReapPolicy,
    aggregate_agent_scores,
    apply_stakes_method,
    optimize_agent_weights,
)
from adaptivepy.stats.cluster_stats import ClusterStats


def _make_frame(
    traj_id: int,
    frame_id: int,
    features: list[float],
    cluster_id: int,
    global_index: int,
) -> FrameRecord:
    return FrameRecord(
        traj_id=traj_id,
        frame_id=frame_id,
        features=np.array(features, dtype=float),
        cluster_id=cluster_id,
        global_index=global_index,
    )


def _make_cluster_stats(
    cluster_features: dict[int, list[list[float]]],
) -> ClusterStats:
    stats: ClusterStats = {}
    global_index = 0
    for cluster_id, feature_rows in cluster_features.items():
        frames = [
            _make_frame(0, frame_id, features, cluster_id, global_index + frame_id)
            for frame_id, features in enumerate(feature_rows)
        ]
        stats[cluster_id] = {"population": len(frames), "frames": frames}
        global_index += len(frames)
    return stats


def _make_cluster_stats_with_trajs(
    cluster_frames: dict[int, list[tuple[int, list[float]]]],
) -> ClusterStats:
    stats: ClusterStats = {}
    global_index = 0
    for cluster_id, frame_rows in cluster_frames.items():
        frames = [
            _make_frame(traj_id, frame_id, features, cluster_id, global_index + frame_id)
            for frame_id, (traj_id, features) in enumerate(frame_rows)
        ]
        stats[cluster_id] = {"population": len(frames), "frames": frames}
        global_index += len(frames)
    return stats


@pytest.fixture
def synthetic_features(tmp_path: Path) -> Path:
    """Create synthetic feature arrays for two trajectories."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    rng = np.random.default_rng(0)
    np.save(features_dir / "traj_0.npy", rng.normal(size=(50, 4)))
    np.save(features_dir / "traj_1.npy", rng.normal(size=(30, 4)))
    return features_dir


def test_list_policies() -> None:
    """Registered policies include all built-in policies."""
    policies = list_policies()
    assert "least_counts" in policies
    assert "random" in policies
    assert "fast" in policies
    assert "ma_reap" in policies
    assert "knn_as" in policies
    assert "maxent_vampnet" in policies


def test_compute_knn_as_scores_vectorsum() -> None:
    """Vector-sum scoring ranks locally asymmetric states highest."""
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ]
    )
    scores, effective_k = compute_knn_as_scores(vectors, k=3, scoring="vectorsum")
    assert effective_k == 3
    assert int(np.argmax(scores)) == 3


def test_compute_knn_as_scores_distance() -> None:
    """Distance scoring ranks isolated states highest."""
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ]
    )
    scores, effective_k = compute_knn_as_scores(vectors, k=3, scoring="distance")
    assert effective_k == 3
    assert int(np.argmax(scores)) == 3


def test_compute_knn_as_scores_clamps_k() -> None:
    """kNN-AS clamps k to the number of available states."""
    vectors = np.array([[0.0], [2.0]])
    scores, effective_k = compute_knn_as_scores(vectors, k=5)
    assert effective_k == 2
    assert scores.shape == (2,)


def test_compute_knn_as_scores_rejects_invalid_params() -> None:
    """Invalid kNN-AS scoring inputs raise clear errors."""
    vectors = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="k"):
        compute_knn_as_scores(vectors, k=1)
    with pytest.raises(ValueError, match="scoring"):
        compute_knn_as_scores(vectors, k=2, scoring="bad")  # type: ignore[arg-type]


def test_knn_as_policy_selects_cluster_ids_and_records_scores() -> None:
    """kNN-AS selects cluster IDs and records score metadata."""
    cluster_stats = _make_cluster_stats(
        {
            10: [[0.0, 0.0]],
            11: [[1.0, 0.0]],
            12: [[0.0, 1.0]],
            13: [[10.0, 10.0]],
        }
    )
    policy = KnnAsPolicy(k=3, scoring="distance")
    selected = policy.select_clusters(cluster_stats, n_seeds=2)
    assert selected[0] == 13
    assert set(selected).issubset(cluster_stats.keys())
    assert policy.last_scores[13]["scoring"] == "distance"
    assert policy.last_scores[13]["effective_k"] == 3


def test_knn_as_policy_tie_breaks_by_population_then_cluster_id() -> None:
    """Equal kNN-AS scores use deterministic population and ID tie-breaks."""
    cluster_stats = _make_cluster_stats(
        {
            0: [[0.0], [0.0]],
            1: [[0.0]],
            2: [[0.0]],
        }
    )
    policy = KnnAsPolicy(k=2)
    selected = policy.select_clusters(cluster_stats, n_seeds=3)
    assert selected == [1, 2, 0]


def test_feature_scale_maximize_and_minimize() -> None:
    """Feature scaling maps values to [0, 1] by direction."""
    values = {0: 1.0, 1: 3.0, 2: 2.0}
    maximize = feature_scale(values, "maximize")
    minimize = feature_scale(values, "minimize")

    assert maximize[0] == pytest.approx(0.0)
    assert maximize[1] == pytest.approx(1.0)
    assert maximize[2] == pytest.approx(0.5)
    assert minimize[0] == pytest.approx(1.0)
    assert minimize[1] == pytest.approx(0.0)


def test_feature_scale_degenerate() -> None:
    """Equal descriptor values yield zero directed contribution."""
    values = {0: 2.0, 1: 2.0}
    scaled = feature_scale(values, "maximize")
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[1] == pytest.approx(0.0)


def test_fast_policy_maximize_selects_balanced_cluster() -> None:
    """FAST maximize favors high feature values and low populations."""
    cluster_stats = _make_cluster_stats(
        {
            0: [[0.0, 0.0]],
            1: [[10.0, 0.0]] * 10,
            2: [[5.0, 0.0]],
        }
    )
    policy = FastPolicy(feature_indices=[0], alpha=1.0)
    selected = policy.select_clusters(cluster_stats, n_seeds=1)
    assert selected == [2]


def test_fast_policy_minimize() -> None:
    """FAST minimize favors clusters with lower feature values."""
    cluster_stats = _make_cluster_stats(
        {
            0: [[0.0]],
            1: [[10.0]] * 5,
        }
    )
    policy = FastPolicy(feature_indices=[0], directions=["minimize"], alpha=0.0)
    selected = policy.select_clusters(cluster_stats, n_seeds=1)
    assert selected == [0]


def test_fast_policy_weighted_features() -> None:
    """Multiple features combine with user-provided weights."""
    cluster_stats = _make_cluster_stats(
        {
            0: [[10.0, 0.0]],
            1: [[0.0, 10.0]],
        }
    )
    policy = FastPolicy(
        feature_indices=[0, 1],
        directions=["maximize", "maximize"],
        weights=[1.0, 0.0],
        alpha=0.0,
    )
    selected = policy.select_clusters(cluster_stats, n_seeds=1)
    assert selected == [0]


def test_fast_policy_equal_populations_exploration_zero() -> None:
    """Equal populations produce zero exploration contribution."""
    cluster_stats = _make_cluster_stats({0: [[1.0]], 1: [[2.0]]})
    _, exploration, _ = compute_fast_rewards(
        cluster_stats, [0], ["maximize"], [1.0], alpha=1.0
    )
    assert exploration[0] == pytest.approx(0.0)
    assert exploration[1] == pytest.approx(0.0)


def test_validate_fast_policy_params() -> None:
    """FAST config validation normalizes indices and defaults."""
    kwargs = validate_fast_policy_params({"feature_indices": [0, 2]}, n_features=4)
    assert kwargs["feature_indices"] == [0, 2]
    assert kwargs["alpha"] == pytest.approx(1.0)
    assert "directions" not in kwargs


def test_validate_knn_as_policy_params_defaults_and_explicit() -> None:
    """kNN-AS config validation normalizes defaults and explicit settings."""
    defaults = validate_knn_as_policy_params({})
    assert defaults == {"k": 5, "scoring": "vectorsum"}

    kwargs = validate_knn_as_policy_params({"k": 3, "scoring": "distance"})
    assert kwargs == {"k": 3, "scoring": "distance"}


def test_validate_knn_as_policy_params_rejects_invalid_values() -> None:
    """kNN-AS config validation rejects invalid k and scoring."""
    with pytest.raises(ValueError, match="k"):
        validate_knn_as_policy_params({"k": 1})
    with pytest.raises(ValueError, match="scoring"):
        validate_knn_as_policy_params({"scoring": "bad"})


def test_validate_fast_policy_params_rejects_invalid_index() -> None:
    """Out-of-range feature indices raise during validation."""
    with pytest.raises(ValueError, match="out of range"):
        validate_fast_policy_params({"feature_indices": [4]}, n_features=4)


def test_load_config_policy_params(tmp_path: Path, synthetic_features: Path) -> None:
    """YAML policy_params are parsed into RunConfig."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {tmp_path / "out"}
policies:
  - fast
policy_params:
  fast:
    feature_indices: [0, 1]
    alpha: 0.5
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.policy_params["fast"]["feature_indices"] == [0, 1]
    assert config.policy_params["fast"]["alpha"] == 0.5


def test_build_policy_kwargs_fast(tmp_path: Path, synthetic_features: Path) -> None:
    """build_policy_kwargs validates FAST settings against feature dimension."""
    config = RunConfig(
        features_dir=synthetic_features,
        output_dir=tmp_path / "out",
        policies=["fast"],
        policy_params={"fast": {"feature_indices": [0]}},
    )
    kwargs = build_policy_kwargs("fast", config, n_features=4)
    assert kwargs["feature_indices"] == [0]


def test_build_policy_kwargs_knn_as(tmp_path: Path, synthetic_features: Path) -> None:
    """build_policy_kwargs validates kNN-AS settings."""
    config = RunConfig(
        features_dir=synthetic_features,
        output_dir=tmp_path / "out",
        policies=["knn_as"],
        policy_params={"knn_as": {"k": 4, "scoring": "distance"}},
    )
    kwargs = build_policy_kwargs("knn_as", config, n_features=4)
    assert kwargs == {"k": 4, "scoring": "distance"}


def test_validate_config_requires_fast_feature_indices(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """validate_config fails when FAST is configured without feature indices."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {tmp_path / "out"}
policies:
  - fast
policy_params:
  fast: {{}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feature_indices"):
        validate_config(config_path)


def test_run_adaptive_sampling_with_fast(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """End-to-end FAST run produces seeds and score artifacts."""
    output_dir = tmp_path / "results_fast"
    config_path = tmp_path / "config_fast.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
clustering:
  method: kmeans
  n_clusters: 3
policies:
  - fast
policy_params:
  fast:
    feature_indices: [0]
    alpha: 1.0
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert "fast" in results
    assert len(results["fast"]) == 2
    assert (output_dir / "fast" / "seeds.csv").is_file()
    assert (output_dir / "fast" / "scores.csv").is_file()


def test_run_adaptive_sampling_with_knn_as(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """End-to-end kNN-AS run produces seeds and score artifacts."""
    output_dir = tmp_path / "results_knn_as"
    config_path = tmp_path / "config_knn_as.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
clustering:
  method: kmeans
  n_clusters: 3
policies:
  - knn_as
policy_params:
  knn_as:
    k: 3
    scoring: vectorsum
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert "knn_as" in results
    assert len(results["knn_as"]) == 2
    assert (output_dir / "knn_as" / "seeds.csv").is_file()
    assert (output_dir / "knn_as" / "scores.csv").is_file()


def test_run_adaptive_sampling(tmp_path: Path, synthetic_features: Path) -> None:
    """End-to-end run produces expected output artifacts."""
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
clustering:
  method: kmeans
  n_clusters: 3
policies:
  - least_counts
  - random
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert len(results) == 2
    assert (output_dir / "assignments.npy").is_file()
    assert (output_dir / "cluster_model.pkl").is_file()
    assert (output_dir / "metadata.csv").is_file()
    assert (output_dir / "run_config.yaml").is_file()
    assert (output_dir / "least_counts" / "seeds.csv").is_file()
    assert (output_dir / "random" / "seeds.csv").is_file()
    assert (output_dir / "combined_metadata.csv").is_file()


def test_load_pkl_features(tmp_path: Path) -> None:
    """Feature loading supports .pkl files with the same shape contract."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    rng = np.random.default_rng(1)
    array = rng.normal(size=(25, 6))
    joblib.dump(array, features_dir / "traj_0.pkl")

    loaded = load_feature_array(features_dir / "traj_0.pkl")
    np.testing.assert_array_equal(loaded, array)

    dataset = load_features(features_dir)
    assert dataset.feature_matrix.shape == (25, 6)
    assert dataset.traj_names == ["traj_0"]


def test_validate_config(tmp_path: Path, synthetic_features: Path) -> None:
    """Validation succeeds for a well-formed configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {tmp_path / "out"}
policies:
  - least_counts
""",
        encoding="utf-8",
    )
    config = validate_config(config_path)
    assert config.features_dir == synthetic_features


def test_apply_stakes_methods() -> None:
    """Stakes methods normalize per candidate column."""
    raw = np.array([[2.0, 0.0], [0.0, 4.0]])
    percentage = apply_stakes_method(raw, "percentage")
    assert percentage[0, 0] == pytest.approx(1.0)
    assert percentage[1, 1] == pytest.approx(1.0)

    equal = apply_stakes_method(np.array([[1.0, 1.0], [1.0, 0.0]]), "equal")
    assert equal[0, 0] == pytest.approx(0.5)
    assert equal[1, 0] == pytest.approx(0.5)
    assert equal[0, 1] == pytest.approx(1.0)

    maximum = apply_stakes_method(np.array([[1.0, 2.0], [3.0, 1.0]]), "max")
    assert maximum[1, 0] == pytest.approx(1.0)
    assert maximum[0, 1] == pytest.approx(1.0)


def test_aggregate_agent_scores_regimes() -> None:
    """Reward aggregation matches collaborative, noncollaborative, competitive."""
    scores = np.array([[1.0, 2.0], [3.0, 1.0]])
    np.testing.assert_array_equal(
        aggregate_agent_scores(scores, "collaborative"), np.array([4.0, 3.0])
    )
    np.testing.assert_array_equal(
        aggregate_agent_scores(scores, "noncollaborative"), np.array([3.0, 2.0])
    )
    np.testing.assert_array_equal(
        aggregate_agent_scores(scores, "competitive"), np.array([2.0, 1.0])
    )


def test_optimize_agent_weights_constraints() -> None:
    """Weight optimization stays on simplex and respects delta."""
    means = np.array([0.0, 0.0])
    stdev = np.array([1.0, 1.0])
    stakes = np.array([1.0, 1.0])
    candidates = np.array([[1.0, 0.0], [0.0, 1.0]])
    prev = np.array([0.5, 0.5])
    updated = optimize_agent_weights(prev, 0.05, means, stdev, stakes, candidates)
    assert np.all(updated >= 0)
    assert updated.sum() == pytest.approx(1.0)
    assert np.all(np.abs(updated - prev) <= 0.05 + 1e-8)


def test_ma_reap_policy_selects_clusters() -> None:
    """MA-REAP returns deterministic cluster selections on synthetic stats."""
    cluster_stats = _make_cluster_stats_with_trajs(
        {
            0: [(0, [0.0, 0.0]), (0, [0.1, 0.0])],
            1: [(1, [5.0, 5.0])],
            2: [(1, [10.0, 0.0]), (1, [10.1, 0.0])],
        }
    )
    centers = np.array([[0.05, 0.0], [5.0, 5.0], [10.05, 0.0]])
    policy = MaReapPolicy(
        agent_assignments={"agent_a": ["traj_0"], "agent_b": ["traj_1"]},
        traj_names=["traj_0", "traj_1"],
        cluster_centers=centers,
        n_candidates=3,
        delta=0.2,
        regime="collaborative",
    )
    selected = policy.select_clusters(cluster_stats, n_seeds=2)
    assert len(selected) == 2
    assert set(selected).issubset({0, 1, 2})
    assert policy.last_weights
    assert policy.last_stakes
    assert policy.last_executors


def test_validate_ma_reap_policy_params() -> None:
    """MA-REAP config validation normalizes agent assignments."""
    kwargs = validate_ma_reap_policy_params(
        {
            "agents": {"agent_a": ["traj_0"], "agent_b": ["traj_1"]},
            "delta": 0.1,
        },
        traj_names=["traj_0", "traj_1"],
        n_features=2,
        n_seeds=2,
        n_clusters=3,
    )
    assert kwargs["agent_assignments"]["agent_a"] == ["traj_0"]
    assert kwargs["n_candidates"] == 3
    assert kwargs["delta"] == pytest.approx(0.1)


def test_validate_ma_reap_rejects_missing_agents() -> None:
    """MA-REAP validation requires agent mapping."""
    with pytest.raises(ValueError, match="agents"):
        validate_ma_reap_policy_params({}, traj_names=["traj_0"], n_features=2)


def test_validate_ma_reap_rejects_unassigned_trajectory() -> None:
    """Every trajectory stem must belong to an agent."""
    with pytest.raises(ValueError, match="Unassigned"):
        validate_ma_reap_policy_params(
            {"agents": {"agent_a": ["traj_0"], "agent_b": ["traj_1"]}},
            traj_names=["traj_0", "traj_1", "traj_2"],
            n_features=2,
        )


def test_build_policy_kwargs_ma_reap(tmp_path: Path, synthetic_features: Path) -> None:
    """build_policy_kwargs validates MA-REAP settings."""
    config = RunConfig(
        features_dir=synthetic_features,
        output_dir=tmp_path / "out",
        policies=["ma_reap"],
        n_seeds=2,
        policy_params={
            "ma_reap": {
                "agents": {"agent_a": ["traj_0"], "agent_b": ["traj_1"]},
            }
        },
    )
    kwargs = build_policy_kwargs(
        "ma_reap",
        config,
        n_features=4,
        traj_names=["traj_0", "traj_1"],
        n_clusters=3,
    )
    assert "agent_assignments" in kwargs


def test_run_adaptive_sampling_with_ma_reap(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """End-to-end MA-REAP run produces seeds and sidecar artifacts."""
    output_dir = tmp_path / "results_ma_reap"
    config_path = tmp_path / "config_ma_reap.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
clustering:
  method: kmeans
  n_clusters: 3
policies:
  - ma_reap
policy_params:
  ma_reap:
    n_candidates: 3
    agents:
      agent_a: [traj_0]
      agent_b: [traj_1]
    delta: 0.05
    regime: collaborative
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert "ma_reap" in results
    assert len(results["ma_reap"]) == 2
    policy_dir = output_dir / "ma_reap"
    assert (policy_dir / "seeds.csv").is_file()
    assert (policy_dir / "scores.csv").is_file()
    assert (policy_dir / "agent_weights.csv").is_file()
    assert (policy_dir / "stakes.csv").is_file()
    assert (policy_dir / "executors.csv").is_file()


def _make_dataset(
    traj_shapes: dict[int, int],
    n_features: int = 2,
) -> Dataset:
    """Build a synthetic dataset with per-trajectory frame counts."""
    frames: list[FrameRecord] = []
    traj_index_map: dict[int, tuple[int, int]] = {}
    global_index = 0
    for traj_id, n_frames in sorted(traj_shapes.items()):
        start = global_index
        for frame_id in range(n_frames):
            frames.append(
                _make_frame(
                    traj_id,
                    frame_id,
                    [float(frame_id), float(traj_id)],
                    None,
                    global_index,
                )
            )
            global_index += 1
        traj_index_map[traj_id] = (start, global_index)
    feature_matrix = np.stack([frame.features for frame in frames], axis=0)
    traj_names = [f"traj_{traj_id}" for traj_id in sorted(traj_shapes)]
    return Dataset(
        frames=frames,
        feature_matrix=feature_matrix,
        traj_index_map=traj_index_map,
        traj_names=traj_names,
    )


class _FakeMaxEntEstimator:
    """Deterministic softmax probabilities for MaxEnt policy tests."""

    def transform(self, features: np.ndarray) -> np.ndarray:
        n_frames = features.shape[0]
        probabilities = np.zeros((n_frames, 2), dtype=float)
        for idx in range(n_frames):
            if idx % 3 == 0:
                probabilities[idx] = [0.5, 0.5]
            else:
                probabilities[idx] = [0.9, 0.1]
        return probabilities


def test_compute_shannon_entropy_uniform_and_peaked() -> None:
    """Shannon entropy is maximal for uniform softmax probabilities."""
    probabilities = np.array([[0.5, 0.5], [0.9, 0.1]])
    entropies = compute_shannon_entropy(probabilities)
    assert entropies[0] > entropies[1]
    assert entropies[0] == pytest.approx(np.log(2.0))


def test_rank_frames_by_entropy_tie_breaks_by_global_index() -> None:
    """Equal entropy values break ties by ascending global index."""
    entropies = np.array([1.0, 1.0, 0.2])
    selected = rank_frames_by_entropy(
        entropies,
        global_indices=[10, 5, 7],
        n_seeds=2,
    )
    assert selected == [1, 0]


def test_split_trajectories_from_dataset_preserves_boundaries() -> None:
    """Trajectory slicing keeps frames grouped by trajectory ID."""
    dataset = _make_dataset({0: 3, 1: 2})
    trajectories = split_trajectories_from_dataset(dataset)
    assert len(trajectories) == 2
    assert trajectories[0].shape == (3, 2)
    assert trajectories[1].shape == (2, 2)
    np.testing.assert_array_equal(trajectories[0][:, 1], [0.0, 0.0, 0.0])


def test_validate_maxent_vampnet_policy_params_defaults() -> None:
    """MaxEnt config validation normalizes defaults."""
    kwargs = validate_maxent_vampnet_policy_params(
        {},
        n_features=4,
        traj_index_map={0: (0, 5), 1: (5, 10)},
    )
    assert kwargs["n_features"] == 4
    assert kwargs["output_states"] == 4
    assert kwargs["lagtime"] == 1
    assert kwargs["batch_size"] == 2048
    assert kwargs["epochs"] == 100


def test_validate_maxent_vampnet_policy_params_rejects_short_trajectory() -> None:
    """Lagtime validation fails when trajectories are too short."""
    with pytest.raises(ValueError, match="lagtime"):
        validate_maxent_vampnet_policy_params(
            {"lagtime": 5},
            n_features=4,
            traj_index_map={0: (0, 4)},
        )


def test_validate_maxent_vampnet_policy_params_rejects_invalid_output_states() -> None:
    """output_states must be at least two."""
    with pytest.raises(ValueError, match="output_states"):
        validate_maxent_vampnet_policy_params({"output_states": 1}, n_features=4)


def test_build_policy_kwargs_maxent_vampnet(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """build_policy_kwargs validates MaxEnt VAMPNet settings."""
    dataset = load_features(synthetic_features)
    config = RunConfig(
        features_dir=synthetic_features,
        output_dir=tmp_path / "out",
        policies=["maxent_vampnet"],
        policy_params={"maxent_vampnet": {"epochs": 2, "output_states": 4}},
    )
    kwargs = build_policy_kwargs(
        "maxent_vampnet",
        config,
        n_features=4,
        traj_index_map=dataset.traj_index_map,
    )
    assert kwargs["epochs"] == 2
    assert kwargs["output_states"] == 4


def test_maxent_vampnet_policy_select_frames_with_fake_estimator() -> None:
    """MaxEnt policy selects highest-entropy frames without clustering."""
    dataset = _make_dataset({0: 4, 1: 4}, n_features=2)
    policy = MaxEntVampNetPolicy(
        n_features=2,
        output_states=2,
        lagtime=1,
        estimator=_FakeMaxEntEstimator(),
    )
    seeds = policy.select_frames(dataset, n_seeds=2)
    assert len(seeds) == 2
    assert seeds[0].policy == "maxent_vampnet"
    assert seeds[0].cluster_id is None
    selected_globals = {seed.global_index for seed in seeds}
    assert selected_globals == {0, 3}
    assert policy.last_scores[0]["entropy"] > policy.last_scores[1]["entropy"]


def test_run_adaptive_sampling_with_maxent_vampnet(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """End-to-end MaxEnt-only run skips clustering and writes score artifacts."""
    pytest.importorskip("deeptime")
    pytest.importorskip("torch")

    output_dir = tmp_path / "results_maxent"
    config_path = tmp_path / "config_maxent.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
policies:
  - maxent_vampnet
policy_params:
  maxent_vampnet:
    output_states: 4
    lagtime: 1
    hidden_layers: [8, 4]
    batch_size: 16
    epochs: 1
    device: cpu
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert "maxent_vampnet" in results
    assert len(results["maxent_vampnet"]) == 2
    assert not (output_dir / "assignments.npy").exists()
    assert not (output_dir / "cluster_model.pkl").exists()
    assert not (output_dir / "metadata.csv").exists()
    policy_dir = output_dir / "maxent_vampnet"
    assert (policy_dir / "seeds.csv").is_file()
    assert (policy_dir / "scores.csv").is_file()
    assert not (policy_dir / "metadata.csv").exists()


def test_run_adaptive_sampling_mixed_maxent_and_cluster_policy(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """Mixed runs cluster once and still score MaxEnt from raw features."""
    pytest.importorskip("deeptime")
    pytest.importorskip("torch")

    output_dir = tmp_path / "results_mixed"
    config_path = tmp_path / "config_mixed.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {output_dir}
clustering:
  method: kmeans
  n_clusters: 3
policies:
  - least_counts
  - maxent_vampnet
policy_params:
  maxent_vampnet:
    output_states: 4
    lagtime: 1
    hidden_layers: [8, 4]
    batch_size: 16
    epochs: 1
n_seeds: 2
random_seed: 7
write_pdbs: false
""",
        encoding="utf-8",
    )

    results = run_adaptive_sampling(config_path)
    assert set(results) == {"least_counts", "maxent_vampnet"}
    assert (output_dir / "assignments.npy").is_file()
    assert (output_dir / "maxent_vampnet" / "scores.csv").is_file()
    assert not (output_dir / "maxent_vampnet" / "metadata.csv").exists()


def test_validate_config_maxent_vampnet(
    tmp_path: Path,
    synthetic_features: Path,
) -> None:
    """validate_config accepts a valid MaxEnt VAMPNet configuration."""
    config_path = tmp_path / "config_maxent_validate.yaml"
    config_path.write_text(
        f"""
features_dir: {synthetic_features}
output_dir: {tmp_path / "out"}
policies:
  - maxent_vampnet
policy_params:
  maxent_vampnet:
    output_states: 4
    lagtime: 1
""",
        encoding="utf-8",
    )
    config = validate_config(config_path)
    assert "maxent_vampnet" in config.policies
