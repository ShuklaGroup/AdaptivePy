"""Coordinate trajectory loading and frame extraction via mdtraj."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import mdtraj as md

from adaptivepy.io.loader import list_trajectory_files

logger = logging.getLogger(__name__)


def build_trajectory_map(
    trajectories_dir: Path,
    traj_names: List[str],
) -> Dict[int, Path]:
    """Map trajectory IDs to coordinate file paths by matching stems.

    Parameters
    ----------
    trajectories_dir : Path
        Directory containing coordinate trajectories.
    traj_names : list of str
        Basenames corresponding to feature files (e.g. ``traj_0``).

    Returns
    -------
    dict
        Mapping from ``traj_id`` to trajectory file path.

    Raises
    ------
    ValueError
        If a trajectory file cannot be found for any ``traj_name``.
    """
    traj_files = list_trajectory_files(trajectories_dir)
    stem_to_path = {path.stem: path for path in traj_files}

    mapping: Dict[int, Path] = {}
    for traj_id, name in enumerate(traj_names):
        if name not in stem_to_path:
            raise ValueError(
                f"No trajectory file found matching feature stem '{name}' "
                f"in {trajectories_dir}"
            )
        mapping[traj_id] = stem_to_path[name]
    return mapping


def load_trajectory(topology: Path, trajectory_path: Path) -> md.Trajectory:
    """Load a single trajectory using mdtraj.

    Parameters
    ----------
    topology : Path
        Topology file (PDB, parm7, etc.).
    trajectory_path : Path
        Coordinate trajectory file.

    Returns
    -------
    mdtraj.Trajectory
        Loaded trajectory object.
    """
    logger.info("Loading trajectory %s with topology %s", trajectory_path, topology)
    return md.load(str(trajectory_path), top=str(topology))


def extract_frame(
    topology: Path,
    trajectory_path: Path,
    frame_id: int,
) -> md.Trajectory:
    """Load a trajectory and return a single-frame subset.

    Parameters
    ----------
    topology : Path
        Topology file path.
    trajectory_path : Path
        Coordinate trajectory file path.
    frame_id : int
        Zero-based frame index to extract.

    Returns
    -------
    mdtraj.Trajectory
        Single-frame trajectory suitable for PDB export.
    """
    traj = load_trajectory(topology, trajectory_path)
    if frame_id < 0 or frame_id >= traj.n_frames:
        raise IndexError(
            f"frame_id {frame_id} out of range for trajectory with "
            f"{traj.n_frames} frames ({trajectory_path})"
        )
    return traj[frame_id]


def get_trajectory_frame_count(topology: Path, trajectory_path: Path) -> int:
    """Return the number of frames in a trajectory without loading all coordinates.

    Parameters
    ----------
    topology : Path
        Topology file path.
    trajectory_path : Path
        Coordinate trajectory file path.

    Returns
    -------
    int
        Number of frames in the trajectory.
    """
    traj = md.load(str(trajectory_path), top=str(topology))
    return traj.n_frames


def validate_trajectory_frame_counts(
    topology: Path,
    trajectory_map: Dict[int, Path],
    expected_counts: Dict[int, int],
) -> None:
    """Verify trajectory frame counts match feature frame counts.

    Parameters
    ----------
    topology : Path
        Topology file path.
    trajectory_map : dict
        Mapping from ``traj_id`` to trajectory file.
    expected_counts : dict
        Expected frame count per ``traj_id`` from features.

    Raises
    ------
    ValueError
        If any trajectory has a different number of frames than its features.
    """
    for traj_id, traj_path in trajectory_map.items():
        n_traj_frames = get_trajectory_frame_count(topology, traj_path)
        n_feature_frames = expected_counts.get(traj_id)
        if n_feature_frames is None:
            continue
        if n_traj_frames != n_feature_frames:
            raise ValueError(
                f"Frame count mismatch for traj_id {traj_id} ({traj_path.name}): "
                f"trajectory has {n_traj_frames}, features have {n_feature_frames}"
            )
