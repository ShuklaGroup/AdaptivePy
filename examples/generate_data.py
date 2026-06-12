#!/usr/bin/env python3
"""Generate synthetic example data for AdaptivePy demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    """Write small random feature arrays under examples/data/features/."""
    root = Path(__file__).resolve().parent
    features_dir = root / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for name, n_frames in [("traj_0", 100), ("traj_1", 80)]:
        features = rng.normal(size=(n_frames, 8)).astype(np.float64)
        np.save(features_dir / f"{name}.npy", features)
        print(f"Wrote {features_dir / name}.npy shape={features.shape}")


if __name__ == "__main__":
    main()
