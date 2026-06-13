# Feature Inputs

AdaptivePy requires **feature-based** trajectory inputs. Coordinate trajectories
are optional and used only for PDB export.

## Directory layout

Place one file per trajectory in `features_dir`:

```text
features/
├── traj_0.npy
├── traj_1.npy
└── traj_2.pkl
```

## Supported formats

| Extension | Loader |
|-----------|--------|
| `.npy` | NumPy |
| `.pkl` | joblib |

Both formats must contain a numeric array convertible to NumPy.

## Shape contract

Each file must be a **2D array** with shape `(n_frames, n_features)`:

- **Rows** — frames within that trajectory
- **Columns** — feature dimensions (e.g. tICA, RMSD, distances)

All trajectories must use the same `n_features`.

## Feature columns for FAST sampling

When using the `fast` policy, `policy_params.fast.feature_indices` refer to
**column indices** in each feature array. For example, in a file with shape
`(100, 8)`, index `0` is the first feature dimension and index `2` is the third.

You can maximize some features and minimize others in the same run by setting
`directions` per index. See [Policies](policies.md) for configuration details.

## Trajectory identity

Each file becomes one trajectory, identified by its filename stem:

| File | `traj_id` | `traj_name` |
|------|-----------|-------------|
| `traj_0.npy` | 0 | `traj_0` |
| `traj_1.pkl` | 1 | `traj_1` |

Files are processed in sorted stem order.

## Matching coordinate trajectories

When `trajectories_dir` is provided, feature and trajectory stems must match:

```text
features/traj_0.npy   ↔   trajectories/traj_0.xtc
features/traj_1.pkl   ↔   trajectories/traj_1.xtc
```

Supported trajectory formats include `.xtc`, `.dcd`, `.trr`, `.nc`, and `.pdb`.

Frame counts in features and trajectories must agree for each `traj_id`.

## Duplicate stems

Having both `traj_0.npy` and `traj_0.pkl` in the same directory raises an error.

## What is not supported (v1)

- A single stacked array with shape `(n_traj, n_frames, n_features)` — use separate
  per-trajectory files instead
- Mixed feature dimensions across trajectories
- Feature files without a matching trajectory when PDB export is requested

## Example: creating features

```python
import numpy as np

# 100 frames, 8 features
features = np.random.randn(100, 8)
np.save("features/traj_0.npy", features)
```

Or with joblib:

```python
import joblib
joblib.dump(features, "features/traj_0.pkl")
```

See `examples/generate_data.py` for a runnable script that creates sample data.
