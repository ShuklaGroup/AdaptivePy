# AdaptivePy

AdaptivePy performs **adaptive sampling** on molecular dynamics trajectories using
clustering-based state space partitioning and policy-driven seed selection.

## Install

```bash
pip install adaptivepy-sampling
```

For Torch-backed policies:

```bash
pip install adaptivepy-sampling[torch]
```

## Quick links

- [Getting Started](getting-started.md) — run your first analysis in minutes
- [Configuration](configuration.md) — YAML options and defaults
- [Feature Inputs](features.md) — supported file formats and layout
- [CLI](cli.md) — command-line usage
- [Python API](python-api.md) — programmatic access
- [Policies](policies.md) — seed selection strategies
- [API Reference](reference/api.md) — full module documentation

## What it does

1. Load per-trajectory feature arrays (`.npy` or `.pkl`)
2. Cluster frames in feature space
3. Apply one or more adaptive policies
4. Select seed frames for new simulations
5. Write metadata, assignments, policy-specific scores, and optional PDB structures

## Example

```bash
adaptivepy run examples/config.yaml
```

See the [Getting Started](getting-started.md) guide for a complete walkthrough.
