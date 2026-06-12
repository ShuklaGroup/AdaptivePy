"""Shared I/O utilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory and all parent directories if they do not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        Resolved path to the created directory.
    """
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def copy_file(src: PathLike, dst: PathLike) -> Path:
    """Copy a file to a destination path, creating parent directories as needed.

    Parameters
    ----------
    src : str or Path
        Source file path.
    dst : str or Path
        Destination file path.

    Returns
    -------
    Path
        Resolved destination path.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    shutil.copy2(src_path, dst_path)
    return dst_path.resolve()
