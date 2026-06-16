# Getting Started

This guide walks through a minimal **feature-only** run using the bundled example
configuration.

## 1. Install AdaptivePy

```bash
pip install adaptivepy-sampling
```

For MaxEnt VAMPNet:

```bash
pip install adaptivepy-sampling[maxent]
```

For local development:

```bash
git clone https://github.com/hnadeem2/AdaptivePy.git
cd AdaptivePy
pip install -e ".[dev,docs]"
```

For MaxEnt development:

```bash
pip install -e ".[dev,docs,maxent]"
```

## 2. Prepare feature files

Create one feature file per trajectory in a directory:

```text
features/
├── traj_0.npy   # shape (n_frames, n_features)
├── traj_1.npy
└── ...
```

Each array must be **2D**: rows are frames, columns are features. See
[Feature Inputs](features.md) for `.pkl` support and validation rules.

Generate synthetic example data:

```bash
python examples/generate_data.py
```

## 3. Create a configuration file

Use `examples/config.yaml` as a template:

```yaml
features_dir: examples/data/features
output_dir: examples/results

clustering:
  method: kmeans
  n_clusters: 5

policies:
  - least_counts
  - random

n_seeds: 3
random_seed: 42
write_pdbs: false
```

## 4. Validate inputs (optional)

```bash
adaptivepy validate examples/config.yaml
```

## 5. Run adaptive sampling

```bash
adaptivepy run examples/config.yaml
```

## 6. Inspect results

After a successful run, `output_dir` contains shared clustering artifacts plus
one subdirectory per policy:

```text
results/
├── assignments.npy
├── cluster_model.pkl
├── metadata.csv
├── run_config.yaml
├── logs.txt
├── least_counts/
│   ├── seeds.csv
│   └── metadata.csv
├── random/
│   ├── seeds.csv
│   └── metadata.csv
└── combined_metadata.csv
```

When using `fast`, `ma_reap`, or `knn_as`, those policies add their own
subdirectories with extra score sidecars. MaxEnt VAMPNet writes per-frame
`scores.csv` and skips clustering when used alone — see [Outputs](outputs.md).

See [Outputs](outputs.md) for a full description of each file.

## Next steps

- Add coordinate trajectories and topology for PDB export — see
  [Configuration](configuration.md)
- Compare policies — see [Policies](policies.md) (`least_counts`, `random`, `fast`, `ma_reap`, `knn_as`, `maxent_vampnet`)
- Use goal-oriented sampling with `fast`, multi-agent `ma_reap`, or entropy-based `maxent_vampnet` — see
  [Configuration](configuration.md#policy-parameters) and [Policies](policies.md)
- Integrate into a pipeline — see [Python API](python-api.md)
