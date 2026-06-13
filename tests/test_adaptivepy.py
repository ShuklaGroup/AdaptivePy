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
)
from adaptivepy.io.loader import load_feature_array, load_features
from adaptivepy.models import FrameRecord
from adaptivepy.policies import list_policies
from adaptivepy.policies.fast import FastPolicy, compute_fast_rewards, feature_scale
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
    """Registered policies include least_counts, random, and fast."""
    policies = list_policies()
    assert "least_counts" in policies
    assert "random" in policies
    assert "fast" in policies


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
