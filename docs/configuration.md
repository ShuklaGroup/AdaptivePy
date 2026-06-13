# Configuration

AdaptivePy runs are driven by a YAML configuration file. The CLI loads and
validates this file before clustering and seed selection.

## Minimal configuration

```yaml
features_dir: path/to/features
output_dir: path/to/results
```

## Full example

```yaml
features_dir: examples/data/features
output_dir: examples/results

# Optional coordinate trajectories for PDB export
trajectories_dir: examples/data/trajectories
topology: examples/data/topology.pdb

clustering:
  method: kmeans
  n_clusters: 10
  params: {}

policies:
  - least_counts
  - random

n_seeds: 10

seed_selection:
  method: nearest_center

random_seed: 42
write_pdbs: true
```

## Options reference

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `features_dir` | yes | — | Directory of per-trajectory `.npy` or `.pkl` feature files |
| `output_dir` | yes | — | Directory where results are written |
| `trajectories_dir` | no | — | Coordinate trajectories (`.xtc`, `.dcd`, `.trr`, …) |
| `topology` | if trajectories | — | Topology file for mdtraj (required with `trajectories_dir`) |
| `clustering.method` | no | `kmeans` | `kmeans`, `minibatch_kmeans`, or `regular_space` |
| `clustering.n_clusters` | no | `10` | Number of clusters |
| `clustering.params` | no | `{}` | Extra arguments passed to the clusterer |
| `policies` | no | `[least_counts]` | List of policy names to evaluate |
| `n_seeds` | no | `10` | Seeds selected per policy |
| `seed_selection.method` | no | `nearest_center` | `nearest_center` or `random_frame` |
| `random_seed` | no | `42` | Global random seed |
| `write_pdbs` | no | `true` | Write PDB files when trajectories are available |

## Clustering methods

### `kmeans`

Standard scikit-learn KMeans. Good default for moderate dataset sizes.

### `minibatch_kmeans`

Mini-batch KMeans for large feature matrices. Lower memory and faster on big runs.

### `regular_space`

Greedy regular-space clustering. Requires `min_dist` in `clustering.params`:

```yaml
clustering:
  method: regular_space
  n_clusters: 20
  params:
    min_dist: 0.5
    max_clusters: 20
```

## Seed selection

Once a policy chooses clusters, one frame is picked from each cluster:

| Method | Behavior |
|--------|----------|
| `nearest_center` | Frame closest to the cluster centroid in feature space |
| `random_frame` | Uniform random frame within the cluster |

## Validation rules

- `features_dir` and `output_dir` are required
- If `trajectories_dir` is set, `topology` must also be set
- Feature and trajectory filenames must match by stem (e.g. `traj_0.npy` ↔ `traj_0.xtc`)
- All feature arrays must share the same number of feature dimensions

Use `adaptivepy validate config.yaml` to check inputs before a full run.

## Reproducibility

Every run saves a copy of the configuration to `output_dir/run_config.yaml`, along
with cluster assignments, the fitted model, and a full log file.
