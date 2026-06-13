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
└── combined_metadata.csv
```

## Listing available policies

```bash
adaptivepy list-policies
```

## Extending policies

New policies register automatically via the `POLICY_REGISTRY`. Subclass `Policy`,
set a unique `name`, implement `select_clusters`, and apply the `@register_policy`
decorator:

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

Import your module before running so the decorator executes. See
[API Reference: Policies](reference/policies.md) for the base class documentation.

## See also

- [Configuration](configuration.md) — set policies and `n_seeds` in YAML
- [Outputs](outputs.md) — seed CSV format
