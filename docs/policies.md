# Policies

Policies decide **which clusters** should contribute seed frames. After a policy
selects clusters, the seed selection module picks one frame from each cluster.

## Built-in policies

### `least_counts`

Selects clusters with the **smallest populations**. Clusters are sorted by
ascending frame count and the first `n_seeds` cluster IDs are chosen.

Best for targeting under-sampled regions of conformational space.

### `random`

Uniformly samples `n_seeds` distinct cluster IDs at random. Respects the global
`random_seed` from the configuration for reproducibility.

### `fast`

Implements **FAST** (Fluctuation Amplification of Specific Traits) from
Zimmerman & Bowman (2015). Balances feature-directed exploitation with
exploration of poorly sampled clusters:

```text
reward = directed_score + alpha * exploration_score
```

- **Directed component** — min-max scales mean feature values per cluster for
  user-selected feature columns, oriented toward maximize or minimize.
- **Exploration component** — favors clusters with smaller populations (same
  least-counts scaling used in the paper).

Configure via `policy_params`:

```yaml
policies:
  - fast

policy_params:
  fast:
    feature_indices: [0, 2]
    directions: [maximize, minimize]
    weights: [0.7, 0.3]
    alpha: 1.0
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `feature_indices` | yes | — | Feature column indices in each `(n_frames, n_features)` file |
| `directions` | no | all `maximize` | `maximize` or `minimize` per feature index |
| `weights` | no | equal | Non-negative weights per feature index |
| `alpha` | no | `1.0` | Exploration weight |

FAST writes an additional `fast/scores.csv` with per-cluster reward components.

### `knn_as`

Implements k-nearest-neighbors adaptive sampling based on
[ERovers/kNN-AS](https://github.com/ERovers/kNN-AS). The upstream algorithm ranks
states by nearest-neighbor geometry; AdaptivePy applies the same score to cluster
representatives because policies select clusters before frame-level seed
selection.

Configure via `policy_params`:

```yaml
policies:
  - knn_as

policy_params:
  knn_as:
    k: 5
    scoring: vectorsum
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `k` | no | `5` | Nearest-neighbor records requested; clamped to available clusters |
| `scoring` | no | `vectorsum` | `vectorsum` for summed displacement magnitude, or `distance` for mean neighbor distance |

kNN-AS writes `knn_as/scores.csv` with per-cluster scores and the effective
neighbor count used for the run.

### `ma_reap`

Implements **MA-REAP** (Multiagent REAP) from Kleiman & Shukla (2022). Extends
REAP with multiple coordinated agents that compartmentalize data and share
information at the clustering step via stakes-weighted rewards.

Pipeline:

1. Select least-count cluster candidates (`n_candidates`).
2. Compute per-agent stakes from frame ownership in each candidate cluster.
3. Optimize each agent's CV weights (SLSQP with simplex and `delta` constraints).
4. Score candidates per agent using weighted standardized L1 distance from the
   agent's mean feature vector.
5. Aggregate scores (`collaborative`, `noncollaborative`, or `competitive`).
6. Select top `n_seeds` clusters.

Configure via `policy_params`:

```yaml
policies:
  - ma_reap

policy_params:
  ma_reap:
    n_candidates: 6
    agents:
      agent_0: [traj_0, traj_1]
      agent_1: [traj_2, traj_3]
    initial_weights: [0.5, 0.5]
    delta: 0.05
    stakes_method: percentage
    regime: collaborative
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `agents` | yes | — | Map agent names to feature file stems |
| `n_candidates` | no | `max(n_seeds, 3*n_seeds)` | Least-count candidates to score |
| `initial_weights` | no | uniform | Shared `(n_features,)` or per-agent `(n_agents, n_features)` |
| `delta` | no | `0.05` | Max per-feature weight change |
| `stakes_method` | no | `percentage` | `percentage`, `equal`, `max`, or `logistic` |
| `stakes_k` | if logistic | — | Logistic steepness parameter |
| `regime` | no | `collaborative` | Reward aggregation mode |

MA-REAP writes sidecar files: `scores.csv`, `agent_weights.csv`, `stakes.csv`,
and `executors.csv`. See [Outputs](outputs.md).

### `maxent_vampnet`

Implements **MaxEnt VAMPNet** from Kleiman & Shukla (2023). Unlike cluster-based
policies, MaxEnt VAMPNet trains a deeptime VAMPNet on lagged trajectory features,
transforms each frame into softmax metastable-state probabilities, and selects
the frames with the highest Shannon entropy. **No clustering step is required.**

Install the optional dependencies first:

```bash
pip install -e ".[torch]"
```

Configure via `policy_params`:

```yaml
policies:
  - maxent_vampnet

policy_params:
  maxent_vampnet:
    n_states: 8
    lagtime: 10
    hidden_layers: [16, 32, 64, 128, 256, 128, 64, 32, 16]
    learning_rate: 1.0e-4
    batch_size: 2048
    epochs: 100
    device: cpu
    num_threads: 1
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `n_states` | no | `n_features` | Number of softmax output nodes |
| `lagtime` | no | `1` | Lag time in frames for VAMPNet training |
| `hidden_layers` | no | author default | Hidden MLP layer widths |
| `learning_rate` | no | `1e-4` | VAMPNet learning rate |
| `batch_size` | no | `2048` | Training batch size |
| `epochs` | no | `100` | Training epochs per run |
| `device` | no | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `num_threads` | no | `1` | CPU threads for PyTorch training |

MaxEnt VAMPNet writes `maxent_vampnet/scores.csv` with per-frame entropy and
softmax probabilities. When it is the only configured policy, clustering
artifacts are skipped entirely.

### `ts_dar`

Implements **TS-DAR** (Transition State identification via Dispersion and
vAriational principle Regularized neural networks) from Liu et al. (2025).
TS-DAR trains a Torch neural network on lagged trajectory features, embeds frames
on a hypersphere, regularizes metastable state centers with VAMP-2 and
dispersion losses, and selects frames with the highest out-of-distribution
(OOD) scores. **No clustering step is required.**

Install the optional Torch dependencies first:

```bash
pip install -e ".[torch]"
```

Configure via `policy_params`:

```yaml
policies:
  - ts_dar

policy_params:
  ts_dar:
    n_states: 4
    latent_dim: 3
    hidden_layers: [128, 64]
    lagtime: 10
    learning_rate: 1.0e-3
    batch_size: 2048
    epochs: 100
    pretrain: 10
    beta: 0.01
    gamma: 1.0
    scaling_temperature: 0.1
    proto_update_factor: 0.5
    optimizer: Adam
    device: cpu
    num_threads: 1
    train_split: 0.9
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `n_states` | no | `min(max(2, n_features), 4)` | Number of metastable states |
| `latent_dim` | no | `2` or `3` | Hyperspherical embedding dimension |
| `hidden_layers` | no | `[128, 64]` | Hidden encoder layer widths |
| `lagtime` | no | `1` | Lag time in frames for transition pairs |
| `learning_rate` | no | `1e-3` | Optimizer learning rate |
| `batch_size` | no | `2048` | Training batch size |
| `epochs` | no | `100` | Training epochs per run |
| `pretrain` | no | `10` | VAMP-2-only warmup epochs |
| `beta` | no | `0.01` | Dispersion loss weight |
| `gamma` | no | `1.0` | Hypersphere radius |
| `scaling_temperature` | no | `0.1` | Dispersion loss temperature |
| `proto_update_factor` | no | `0.5` | EMA update factor for state centers |
| `optimizer` | no | `Adam` | `Adam`, `SGD`, or `RMSprop` |
| `device` | no | `cpu` | PyTorch device |
| `num_threads` | no | `1` | CPU threads for PyTorch |
| `train_split` | no | `0.9` | Fraction of lagged pairs used for training |

TS-DAR writes `ts_dar/scores.csv` with per-frame OOD scores, assigned states,
hyperspherical embeddings, and softmax probabilities. When it is the only
configured policy, clustering artifacts are skipped entirely.

## Multi-policy runs

Configure multiple policies in YAML:

```yaml
policies:
  - least_counts
  - random
```

Each policy writes results to its own subdirectory under `output_dir`:

```text
results/
├── least_counts/
│   ├── seeds.csv
│   └── metadata.csv
├── random/
│   ├── seeds.csv
│   └── metadata.csv
├── fast/
│   ├── seeds.csv
│   ├── metadata.csv
│   └── scores.csv
├── knn_as/
│   ├── seeds.csv
│   ├── metadata.csv
│   └── scores.csv
├── ma_reap/
│   ├── seeds.csv
│   ├── metadata.csv
│   ├── scores.csv
│   ├── agent_weights.csv
│   ├── stakes.csv
│   └── executors.csv
├── maxent_vampnet/
│   ├── seeds.csv
│   └── scores.csv
├── ts_dar/
│   ├── seeds.csv
│   └── scores.csv
├── metapolicy/
│   ├── seeds.csv
│   ├── metadata.csv
│   └── votes.csv
└── combined_metadata.csv
```

## Metapolicy ensembles

Metapolicies are opt-in ensembles over cluster-level rankings from multiple
policies. They do not replace individual policy outputs; they add a final
ensemble seed set under `metapolicy/`.

```yaml
metapolicy:
  enabled: true
  name: ensemble
  strategy: majority_polling
  policies: [least_counts, random, fast]
  n_seeds: 10
```

`majority_polling` selects clusters by policy vote count, weighted rank score,
smaller cluster population, then cluster ID. `allocation` uses fixed per-policy
quotas:

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

When MaxEnt VAMPNet or TS-DAR participates in an ensemble, AdaptivePy clusters
the dataset and converts frame scores to a cluster ranking by taking the maximum
entropy or OOD score among frames in each cluster.

## Listing available policies

```bash
adaptivepy list-policies
```

## Extending policies

New policies register automatically via the `POLICY_REGISTRY`. Subclass `Policy`,
set a unique `name`, and apply the `@register_policy` decorator.

Cluster-based policies set `requires_clustering = True` (default) and implement
`select_clusters`. Frame-level policies set `requires_clustering = False` and
implement `select_frames`:

```python
from adaptivepy.policies.base import Policy, register_policy
from adaptivepy.stats.cluster_stats import ClusterStats


@register_policy
class MyPolicy(Policy):
    name = "my_policy"

    def select_clusters(self, cluster_stats: ClusterStats, n_seeds: int):
        # Return a list of cluster IDs
        ...
```

```python
from adaptivepy.models import Dataset, SeedResult
from adaptivepy.policies.base import Policy, register_policy


@register_policy
class MyFramePolicy(Policy):
    name = "my_frame_policy"
    requires_clustering = False

    def select_clusters(self, cluster_stats, n_seeds):
        raise NotImplementedError

    def select_frames(self, dataset: Dataset, n_seeds: int) -> list[SeedResult]:
        ...
```

Import your module before running so the decorator executes. See
[API Reference: Policies](reference/policies.md) for the base class documentation.

## See also

- [Configuration](configuration.md) — set policies and `n_seeds` in YAML
- [Outputs](outputs.md) — seed CSV format
