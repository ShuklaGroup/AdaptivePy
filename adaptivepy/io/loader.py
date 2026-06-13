"""Feature loading and dataset validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from adaptivepy.models import Dataset, FrameRecord

logger = logging.getLogger(__name__)

FEATURE_EXTENSIONS = (".npy", ".pkl")


def list_feature_files(features_dir: Path) -> List[Path]:
    """List ``*.npy`` and ``*.pkl`` feature files in a directory, sorted by stem.

    Parameters
    ----------
    features_dir : Path
        Directory containing feature arrays.

    Returns
    -------
    list of Path
        Sorted paths to feature files (one file per trajectory stem).

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no supported feature files are found or duplicate stems exist.
    """
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    files_by_stem: Dict[str, Path] = {}
    for path in sorted(features_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in FEATURE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in files_by_stem:
            raise ValueError(
                f"Duplicate feature files for '{stem}': "
                f"{files_by_stem[stem].name} and {path.name}"
            )
        files_by_stem[stem] = path

    files = [files_by_stem[stem] for stem in sorted(files_by_stem)]
    if not files:
        raise ValueError(
            f"No .npy or .pkl feature files found in {features_dir}"
        )
    return files


def load_feature_array(path: Path) -> np.ndarray:
    """Load a feature array from a ``.npy`` or ``.pkl`` file.

    Parameters
    ----------
    path : Path
        Path to a feature file.

    Returns
    -------
    np.ndarray
        Loaded feature array.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the loaded object cannot be
        converted to an array.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".pkl":
        data = joblib.load(path)
    else:
        raise ValueError(
            f"Unsupported feature file extension '{suffix}' in {path}. "
            f"Use one of: {', '.join(FEATURE_EXTENSIONS)}"
        )

    try:
        return np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Feature file {path} does not contain a numeric array."
        ) from exc


def list_trajectory_files(trajectories_dir: Path) -> List[Path]:
    """List coordinate trajectory files supported by mdtraj.

    Parameters
    ----------
    trajectories_dir : Path
        Directory containing trajectory files.

    Returns
    -------
    list of Path
        Sorted paths to trajectory files (``.xtc``, ``.dcd``, ``.trr``).

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no supported trajectory files are found.
    """
    trajectories_dir = Path(trajectories_dir)
    if not trajectories_dir.is_dir():
        raise FileNotFoundError(
            f"Trajectories directory not found: {trajectories_dir}"
        )

    extensions = ("*.xtc", "*.dcd", "*.trr", "*.nc", "*.pdb")
    files: List[Path] = []
    for pattern in extensions:
        files.extend(trajectories_dir.glob(pattern))
    files = sorted(set(files), key=lambda p: p.name)

    if not files:
        raise ValueError(
            f"No supported trajectory files found in {trajectories_dir}"
        )
    return files


def _stem(path: Path) -> str:
    """Return the filename stem without extension."""
    return path.stem


def validate_feature_trajectory_mapping(
    feature_files: List[Path],
    trajectory_files: Optional[List[Path]] = None,
) -> None:
    """Ensure feature and trajectory filenames match one-to-one.

    Parameters
    ----------
    feature_files : list of Path
        Feature ``.npy`` or ``.pkl`` file paths.
    trajectory_files : list of Path or None
        Optional coordinate trajectory file paths.

    Raises
    ------
    ValueError
        If stems do not match exactly between features and trajectories.
    """
    if trajectory_files is None:
        return

    feature_stems = {_stem(f) for f in feature_files}
    traj_stems = {_stem(t) for t in trajectory_files}

    missing_traj = feature_stems - traj_stems
    missing_features = traj_stems - feature_stems

    if missing_traj or missing_features:
        messages = []
        if missing_traj:
            messages.append(
                f"Features without matching trajectories: {sorted(missing_traj)}"
            )
        if missing_features:
            messages.append(
                f"Trajectories without matching features: {sorted(missing_features)}"
            )
        raise ValueError("; ".join(messages))


def load_features(features_dir: Path) -> Dataset:
    """Load feature arrays from disk and build a :class:`Dataset`.

    Each ``*.npy`` or ``*.pkl`` file must have shape ``(n_frames, n_features)``.
    Per-frame ``FrameRecord`` objects are created while preserving trajectory
    identity.

    Parameters
    ----------
    features_dir : Path
        Directory containing feature files.

    Returns
    -------
    Dataset
        Loaded dataset with concatenated feature matrix and frame records.

    Raises
    ------
    ValueError
        If feature arrays have inconsistent dimensionality.
    """
    feature_files = list_feature_files(features_dir)
    frames: List[FrameRecord] = []
    feature_blocks: List[np.ndarray] = []
    traj_index_map: Dict[int, Tuple[int, int]] = {}
    traj_names: List[str] = []
    global_offset = 0
    n_features: Optional[int] = None

    for traj_id, feature_path in enumerate(feature_files):
        features = load_feature_array(feature_path)
        if features.ndim != 2:
            raise ValueError(
                f"Feature file {feature_path} must be 2D (n_frames, n_features), "
                f"got shape {features.shape}"
            )

        if n_features is None:
            n_features = features.shape[1]
        elif features.shape[1] != n_features:
            raise ValueError(
                f"Inconsistent feature dimension in {feature_path}: "
                f"expected {n_features}, got {features.shape[1]}"
            )

        n_frames = features.shape[0]
        start_idx = global_offset
        end_idx = global_offset + n_frames

        for frame_id in range(n_frames):
            global_index = global_offset + frame_id
            frames.append(
                FrameRecord(
                    traj_id=traj_id,
                    frame_id=frame_id,
                    features=features[frame_id],
                    global_index=global_index,
                )
            )

        feature_blocks.append(features)
        traj_index_map[traj_id] = (start_idx, end_idx)
        traj_names.append(_stem(feature_path))
        global_offset = end_idx

        logger.info(
            "Loaded %s: %d frames, %d features",
            feature_path.name,
            n_frames,
            n_features,
        )

    feature_matrix = (
        np.vstack(feature_blocks) if feature_blocks else np.empty((0, 0))
    )

    return Dataset(
        frames=frames,
        feature_matrix=feature_matrix,
        traj_index_map=traj_index_map,
        traj_names=traj_names,
    )


def validate_dataset(
    dataset: Dataset,
    trajectory_files: Optional[List[Path]] = None,
) -> None:
    """Run consistency checks on a loaded dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to validate.
    trajectory_files : list of Path or None
        Optional trajectory files for cross-validation.

    Raises
    ------
    ValueError
        If internal consistency checks fail.
    """
    if dataset.feature_matrix is None or len(dataset.frames) == 0:
        raise ValueError("Dataset is empty.")

    n_frames, n_features = dataset.feature_matrix.shape
    if n_frames != len(dataset.frames):
        raise ValueError(
            "Feature matrix row count does not match number of frame records."
        )

    for record in dataset.frames:
        if record.features.shape != (n_features,):
            raise ValueError(
                f"Frame ({record.traj_id}, {record.frame_id}) has invalid "
                f"feature shape {record.features.shape}."
            )

    if trajectory_files is not None:
        feature_stems = set(dataset.traj_names)
        traj_stems = {path.stem for path in trajectory_files}
        missing_traj = feature_stems - traj_stems
        missing_features = traj_stems - feature_stems
        if missing_traj or missing_features:
            messages = []
            if missing_traj:
                messages.append(
                    f"Features without matching trajectories: {sorted(missing_traj)}"
                )
            if missing_features:
                messages.append(
                    f"Trajectories without matching features: {sorted(missing_features)}"
                )
            raise ValueError("; ".join(messages))
        if len(trajectory_files) != len(dataset.traj_names):
            raise ValueError(
                "Number of trajectory files must match number of feature files."
            )
