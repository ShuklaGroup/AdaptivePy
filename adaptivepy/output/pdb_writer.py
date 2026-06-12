"""PDB export for selected seed frames using mdtraj."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from adaptivepy.io.trajectory import extract_frame
from adaptivepy.models import SeedResult
from adaptivepy.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


def write_seed_pdbs(
    seeds: List[SeedResult],
    topology: Path,
    trajectory_map: Dict[int, Path],
    output_dir: Path,
) -> List[Path]:
    """Extract coordinate frames and save each seed as a PDB file.

    Parameters
    ----------
    seeds : list of SeedResult
        Selected seed frames.
    topology : Path
        Topology file for mdtraj loading.
    trajectory_map : dict
        Mapping from ``traj_id`` to trajectory file path.
    output_dir : Path
        Directory where PDB files are written (typically ``pdbs/``).

    Returns
    -------
    list of Path
        Paths to written PDB files.
    """
    pdb_dir = ensure_dir(output_dir / "pdbs")
    written: List[Path] = []

    for seed in seeds:
        traj_path = trajectory_map[seed.traj_id]
        frame = extract_frame(topology, traj_path, seed.frame_id)
        pdb_path = pdb_dir / (
            f"seed_{seed.seed_id}_traj{seed.traj_id}_frame{seed.frame_id}.pdb"
        )
        frame.save_pdb(str(pdb_path))
        written.append(pdb_path)
        logger.info(
            "Wrote PDB for seed %d: traj=%d frame=%d -> %s",
            seed.seed_id,
            seed.traj_id,
            seed.frame_id,
            pdb_path.name,
        )

    return written
