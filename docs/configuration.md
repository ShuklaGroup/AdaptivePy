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
| `policy_params` | no | `{}` | Per-policy settings (see below) |
| `metapolicy` | no | disabled | Optional ensemble over multiple policies |
| `n_seeds` | no | `10` | Seeds selected per policy |
| `seed_selection.method` | no | `nearest_center` | `nearest_center` or `random_frame` |
| `random_seed` | no | `42` | Global random seed |
| `write_pdbs` | no | `true` | Write PDB files when trajectories are available |

## Policy parameters

Some policies accept extra settings under `policy_params`:

```yaml
policies:
  - fast
  - ma_reap
  - knn_as
  - least_counts

policy_params:
  fast:
    feature_indices: [0, 2]
    directions: [maximize, minimize]
    weights: [0.7, 0.3]
    alpha: 1.0
  ma_reap:
    n_candidates: 12
    agents:
      agent_0: [traj_0, traj_1]
      agent_1: [traj_2, traj_3]
    delta: 0.05
    stakes_method: percentage
    regime: collaborative
  knn_as:
    k: 5
    scoring: vectorsum
  maxent_vampnet:
    n_states: 8
    lagtime: 10
    epochs: 100
  ts_dar:
    n_states: 4
    latent_dim: 3
    hidden_layers: [128, 64]
    lagtime: 10
    epochs: 100
```

Install Torch-backed policy dependencies with `pip install -e ".[torch]"` when
using `maxent_vampnet` or `ts_dar`.

### FAST (`fast`)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `feature_indices` | yes | — | Feature column indices to optimize |
| `directions` | no | all `maximize` | `maximize` or `minimize` per feature |
| `weights` | no | equal | Non-negative weights per feature |
| `alpha` | no | `1.0` | Exploration/exploitation balance |

See [Policies](policies.md) for algorithm details.

### MA-REAP (`ma_reap`)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `agents` | yes | — | Map agent names to feature file stems |
| `n_candidates` | no | `max(n_seeds, 3*n_seeds)` | Least-count candidates to score |
| `initial_weights` | no | uniform | CV weights, shared or per-agent |
| `delta` | no | `0.05` | Max per-feature weight update |
| `stakes_method` | no | `percentage` | `percentage`, `equal`, `max`, `logistic` |
| `stakes_k` | if logistic | — | Logistic steepness |
| `regime` | no | `collaborative` | `collaborative`, `noncollaborative`, `competitive` |

Every feature trajectory must be assigned to exactly one agent. At least two
agents are required.

### kNN-AS (`knn_as`)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `k` | no | `5` | Nearest-neighbor records requested; clamped to available clusters |
| `scoring` | no | `vectorsum` | `vectorsum` for summed displacement magnitude, or `distance` for mean neighbor distance |

### MaxEnt VAMPNet (`maxent_vampnet`)

Requires `pip install -e ".[torch]"`.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `n_states` | no | `n_features` | Number of softmax output nodes |
| `lagtime` | no | `1` | Lag time in frames for VAMPNet training |
| `hidden_layers` | no | author default | Hidden MLP layer widths |
| `learning_rate` | no | `1e-4` | VAMPNet learning rate |
| `batch_size` | no | `2048` | Training batch size |
| `epochs` | no | `100` | Training epochs per run |
| `device` | no | `cpu` | PyTorch device |
| `num_threads` | no | `1` | CPU threads for PyTorch |

Each trajectory must contain more frames than `lagtime`. When `maxent_vampnet`
is the only configured policy, clustering is skipped.

### TS-DAR (`ts_dar`)

Requires `pip install -e ".[torch]"`.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `n_states` | no | `min(max(2, n_features), 4)` | Number of metastable states |
| `latent_dim` | no | `2` if `n_states <= 3`, else `3` | Hyperspherical embedding dimension |
| `hidden_layers` | no | `[128, 64]` | Hidden encoder layer widths |
| `lagtime` | no | `1` | Lag time in frames for transition pairs |
| `learning_rate` | no | `1e-3` | Optimizer learning rate |
| `batch_size` | no | `2048` | Training batch size |
| `epochs` | no | `100` | Training epochs per run |
| `pretrain` | no | `10` | Epochs trained with VAMP-2 loss only before adding dispersion loss |
| `beta` | no | `0.01` | Dispersion loss weight |
| `gamma` | no | `1.0` | Hypersphere radius |
| `scaling_temperature` | no | `0.1` | Dispersion loss temperature |
| `proto_update_factor` | no | `0.5` | EMA update factor for state centers |
| `optimizer` | no | `Adam` | `Adam`, `SGD`, or `RMSprop` |
| `device` | no | `cpu` | PyTorch device |
| `num_threads` | no | `1` | CPU threads for PyTorch |
| `train_split` | no | `0.9` | Fraction of lagged pairs used for training |

Each trajectory must contain more frames than `lagtime`, and the dataset must
provide at least two lagged frame pairs. When `ts_dar` is the only configured
policy, clustering is skipped.

## Metapolicy ensembles

Set `metapolicy.enabled: true` to combine policy rankings into one final seed set
written under `output_dir/metapolicy/`. Individual policy outputs are still
written as usual.

Majority polling ranks clusters by vote count, weighted rank score, smaller
population, then cluster ID:

```yaml
metapolicy:
  enabled: true
  name: ensemble
  strategy: majority_polling
  policies: [least_counts, random, fast]
  n_seeds: 10
  rank_depth: 10
  weights:
    fast: 2.0
```

Allocation takes a fixed number of seeds from each policy ranking and skips
duplicates by continuing down that policy's ranking:

```yaml
metapolicy:
  enabled: true
  strategy: allocation
  policies: [least_counts, random, fast, knn_as]
  allocations:
    least_counts: 3
    random: 2
    fast: 3
    knn_as: 2
```

If `maxent_vampnet` or `ts_dar` participates in a metapolicy, clustering is
forced so frame scores can be aggregated to clusters. The cluster score is the
maximum entropy or OOD score among frames assigned to that cluster.

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
