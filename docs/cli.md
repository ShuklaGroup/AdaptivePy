# CLI

AdaptivePy provides a Click-based command-line interface installed as `adaptivepy`.

## Commands

### `run`

Execute a full adaptive sampling workflow from a YAML configuration:

```bash
adaptivepy run config.yaml
```

This loads features, clusters frames, applies all configured policies, selects
seeds, and writes outputs to `output_dir`.

### `validate`

Check configuration and input data without running clustering:

```bash
adaptivepy validate config.yaml
```

Useful for catching missing files, shape mismatches, or invalid policy names
before a long run.

### `list-policies`

Print all registered policy names:

```bash
adaptivepy list-policies
```

Example output:

```text
fast
knn_as
least_counts
ma_reap
maxent_vampnet
random
ts_dar
```

Built-in policies:

| Policy | Description |
|--------|-------------|
| `least_counts` | Select least-populated clusters |
| `random` | Random cluster sampling |
| `fast` | Feature-directed exploration/exploitation (Zimmerman & Bowman 2015) |
| `ma_reap` | Multiagent REAP with agent stakes and learned CV weights (Kleiman & Shukla 2022) |
| `knn_as` | k-nearest-neighbors adaptive sampling over cluster representatives (Rovers et al. 2025) |
| `maxent_vampnet` | Entropy-based frame selection via VAMPNet; no clustering (Kleiman & Shukla 2023) |
| `ts_dar` | OOD-score frame selection via TS-DAR hyperspherical embeddings (Liu et al. 2025) |

See [Policies](policies.md) for configuration details.

## Version

```bash
adaptivepy --version
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation or runtime error |

Errors are printed to stderr with a short message.

## Examples

Feature-only run:

```bash
adaptivepy run examples/config.yaml
```

Validate before running:

```bash
adaptivepy validate examples/config.yaml && adaptivepy run examples/config.yaml
```
