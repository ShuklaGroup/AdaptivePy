# AdaptivePy

**Adaptive sampling for molecular dynamics trajectories**

Clustering-based and frame-level adaptive policies for MD workflows, including entropy-based MaxEnt VAMPNet and OOD-based TS-DAR seed selection.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge)](https://shuklagroup.github.io/AdaptivePy/)
[![PyPI](https://img.shields.io/badge/PyPI-adaptivepy--sampling-orange?style=for-the-badge)](https://pypi.org/project/adaptivepy-sampling/)
[![Python](https://img.shields.io/badge/python-3.9+-green?style=for-the-badge)](https://www.python.org/)

---

## Overview

AdaptivePy helps you identify under-sampled or high-uncertainty regions of conformational space and select seed frames for new simulations. It loads per-trajectory feature arrays, optionally clusters frames, applies adaptive policies, and writes reproducible metadata and optional PDB structures.

Most policies select seeds from clusters. **MaxEnt VAMPNet** (`maxent_vampnet`) and **TS-DAR** (`ts_dar`) are frame-level: they train Torch models on lagged features and select scored frames directly — no clustering required.

**Full documentation:** [https://shuklagroup.github.io/AdaptivePy/](https://shuklagroup.github.io/AdaptivePy/)

| | |
|---|---|
| **Input** | Feature arrays (`.npy` / `.pkl`), optional coordinate trajectories |
| **Clustering** | KMeans, MiniBatch KMeans, regular-space (optional for frame-level policies) |
| **Policies** | Least counts, random, FAST, MA-REAP, kNN-AS, MaxEnt VAMPNet, TS-DAR (extensible) |
| **Output** | Seeds, cluster assignments, model, logs, policy scores, optional PDBs |

## Installation

```bash
pip install adaptivepy-sampling
```

For Torch-backed policies such as **MaxEnt VAMPNet** and **TS-DAR**:

```bash
pip install adaptivepy-sampling[torch]
```

For development:

```bash
git clone https://github.com/shuklagroup/AdaptivePy.git
cd AdaptivePy
pip install -e ".[dev,docs]"
```

For Torch-backed policy development:

```bash
pip install -e ".[dev,docs,torch]"
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

See the [Getting Started guide](https://shuklagroup.github.io/AdaptivePy/getting-started/) for a complete walkthrough.

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
| `knn_as` | k-nearest-neighbors adaptive sampling over cluster representatives (Rovers et al. 2025) |
| `maxent_vampnet` | Entropy-based frame selection via VAMPNet soft state assignments (Kleiman & Shukla 2023); no clustering (forced for ensemble policy) |
| `ts_dar` | OOD-score frame selection via TS-DAR hyperspherical embeddings (Liu et al. 2025); no clustering (forced for ensemble policy) |

`fast`, `ma_reap`, `knn_as`, `maxent_vampnet`, and `ts_dar` accept extra YAML under `policy_params`.
MA-REAP requires mapping each trajectory to an agent. Torch-backed policies require the
`[torch]` install extra (`pip install adaptivepy-sampling[torch]`). See the
[Policies guide](https://shuklagroup.github.io/AdaptivePy/policies/) and
[Configuration](https://shuklagroup.github.io/AdaptivePy/configuration/).

AdaptivePy also supports opt-in metapolicy ensembles with majority polling or
per-policy seed allocation through the `metapolicy` YAML block.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](https://shuklagroup.github.io/AdaptivePy/getting-started/) | First run in minutes |
| [Configuration](https://shuklagroup.github.io/AdaptivePy/configuration/) | YAML options and defaults |
| [Feature Inputs](https://shuklagroup.github.io/AdaptivePy/features/) | File formats and layout |
| [Policies](https://shuklagroup.github.io/AdaptivePy/policies/) | Seed selection strategies |
| [API Reference](https://shuklagroup.github.io/AdaptivePy/reference/api/) | Module documentation |

## Contributors

- Hassan Nadeem
- Diego E. Kleiman

## License

MIT. See [LICENSE](LICENSE) for details.
