# AdaptivePy

Adaptive sampling on molecular dynamics trajectories using clustering-based state space partitioning and policy-driven seed selection.

**Documentation:** [https://hnadeem2.github.io/AdaptivePy/](https://hnadeem2.github.io/AdaptivePy/)

## Installation

```bash
pip install adaptivepy-sampling
```

## Quick start

1. Prepare feature files (`features/traj_0.npy` or `traj_0.pkl`, ...) with shape `(n_frames, n_features)`.
2. Optionally add matching coordinate trajectories (`trajectories/traj_0.xtc`, ...) and a topology file.
3. Edit `examples/config.yaml` and run:

```bash
adaptivepy run examples/config.yaml
```

## CLI

```bash
adaptivepy run config.yaml
adaptivepy validate config.yaml
adaptivepy list-policies
```

## Python API

```python
from adaptivepy import run_adaptive_sampling

results = run_adaptive_sampling("config.yaml")
```
