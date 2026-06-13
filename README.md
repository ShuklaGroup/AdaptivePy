# AdaptivePy

**Adaptive sampling for molecular dynamics trajectories**

Clustering-based state space partitioning and policy-driven seed selection for MD workflows.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge)](https://hnadeem2.github.io/AdaptivePy/)
[![PyPI](https://img.shields.io/badge/PyPI-adaptivepy--sampling-orange?style=for-the-badge)](https://pypi.org/project/adaptivepy-sampling/)
[![Python](https://img.shields.io/badge/python-3.9+-green?style=for-the-badge)](https://www.python.org/)

---

## Overview

AdaptivePy helps you identify under-sampled regions of conformational space and select seed frames for new simulations. It loads per-trajectory feature arrays, clusters frames, applies adaptive policies, and writes reproducible metadata and optional PDB structures.

**Full documentation:** [https://hnadeem2.github.io/AdaptivePy/](https://hnadeem2.github.io/AdaptivePy/)

| | |
|---|---|
| **Input** | Feature arrays (`.npy` / `.pkl`), optional coordinate trajectories |
| **Clustering** | KMeans, MiniBatch KMeans, regular-space |
| **Policies** | Least counts, random, FAST, MA-REAP (extensible) |
| **Output** | Seeds, cluster assignments, model, logs, optional PDBs |

## Installation

```bash
pip install adaptivepy-sampling
```

For development:

```bash
git clone https://github.com/hnadeem2/AdaptivePy.git
cd AdaptivePy
pip install -e ".[dev,docs]"
```

## Quick start

1. **Prepare features** — one file per trajectory, shape `(n_frames, n_features)`:

   ```text
   features/
   ├── traj_0.npy
   └── traj_1.pkl
   ```

2. **Configure** — edit `examples/config.yaml` (or create your own).

3. **Run**:

   ```bash
   adaptivepy run examples/config.yaml
   ```

See the [Getting Started guide](https://hnadeem2.github.io/AdaptivePy/getting-started/) for a complete walkthrough.

## CLI

```bash
adaptivepy run config.yaml       # run adaptive sampling
adaptivepy validate config.yaml  # validate inputs only
adaptivepy list-policies         # list available policies
```

## Python API

```python
from adaptivepy import run_adaptive_sampling

results = run_adaptive_sampling("config.yaml")
```

## Policies

Built-in seed-selection policies:

| Policy | Use case |
|--------|----------|
| `least_counts` | Target under-sampled clusters |
| `random` | Baseline random sampling |
| `fast` | Goal-directed sampling via feature columns (Zimmerman & Bowman 2015) |
| `ma_reap` | Multi-agent coordinated sampling with learned CV weights (Kleiman & Shukla 2022) |

`fast` and `ma_reap` accept extra YAML under `policy_params`. MA-REAP requires
mapping each trajectory to an agent. See the
[Policies guide](https://hnadeem2.github.io/AdaptivePy/policies/) and
[Configuration](https://hnadeem2.github.io/AdaptivePy/configuration/).

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](https://hnadeem2.github.io/AdaptivePy/getting-started/) | First run in minutes |
| [Configuration](https://hnadeem2.github.io/AdaptivePy/configuration/) | YAML options and defaults |
| [Feature Inputs](https://hnadeem2.github.io/AdaptivePy/features/) | File formats and layout |
| [Policies](https://hnadeem2.github.io/AdaptivePy/policies/) | Seed selection strategies |
| [API Reference](https://hnadeem2.github.io/AdaptivePy/reference/api/) | Module documentation |

## Contributors

- Hassan

## License

MIT. See [LICENSE](LICENSE) for details.
