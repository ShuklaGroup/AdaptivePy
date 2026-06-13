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
