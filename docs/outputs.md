# Outputs

AdaptivePy writes structured outputs to `output_dir` for reproducibility and
downstream analysis.

## Top-level artifacts

| File | Description |
|------|-------------|
| `run_config.yaml` | Exact copy of the run configuration |
| `assignments.npy` | Per-frame cluster labels (1D integer array) |
| `cluster_model.pkl` | Serialized clustering model (joblib) |
| `metadata.csv` | Cluster populations (`cluster_id`, `population`) |
| `logs.txt` | Full run log |
| `combined_metadata.csv` | Seeds from all policies (multi-policy runs only) |

## Per-policy outputs

Each policy gets a subdirectory named after the policy:

```text
results/least_counts/
├── seeds.csv
├── metadata.csv
└── pdbs/          # optional, when trajectories are provided
    └── seed_0_traj0_frame42.pdb
```

### `seeds.csv`

| Column | Description |
|--------|-------------|
| `seed_id` | Sequential ID within the policy |
| `policy` | Policy name |
| `traj_id` | Source trajectory index |
| `frame_id` | Frame index within the trajectory |
| `cluster_id` | Cluster the seed was drawn from |
| `global_index` | Row index in the concatenated feature matrix |

Example:

```csv
seed_id,policy,traj_id,frame_id,cluster_id,global_index
0,least_counts,0,42,3,42
1,least_counts,1,15,1,65
```

### `metadata.csv`

Cluster population statistics (same format as the top-level file):

```csv
cluster_id,population
0,123
1,45
2,67
```

### MA-REAP sidecar files

When using the `ma_reap` policy, additional CSV files are written:

| File | Description |
|------|-------------|
| `scores.csv` | Per-candidate aggregate and per-agent scores |
| `agent_weights.csv` | Learned CV weights per agent and feature index |
| `stakes.csv` | Agent stakes per candidate cluster |
| `executors.csv` | Which agent executes each selected seed |

## PDB export

When `trajectories_dir`, `topology`, and `write_pdbs: true` are set, selected seed
frames are extracted with mdtraj and saved as PDB files under `pdbs/`.

Filename pattern:

```text
seed_{seed_id}_traj{traj_id}_frame{frame_id}.pdb
```

## Reproducibility

Every run preserves:

- Configuration snapshot
- Random seed
- Cluster model and assignments
- Full metadata and logs

Re-run with the same config and seed to reproduce identical cluster assignments
(for deterministic clustering methods).

## See also

- [Getting Started](getting-started.md) — example output tree
- [Configuration](configuration.md) — control `write_pdbs` and output paths
