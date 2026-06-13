"""Integration tests for AdaptivePy."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from adaptivepy.api import run_adaptive_sampling, validate_config
from adaptivepy.io.loader import load_feature_array, load_features
from adaptivepy.policies import list_policies


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
    """Registered policies include least_counts and random."""
    policies = list_policies()
    assert "least_counts" in policies
    assert "random" in policies


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
