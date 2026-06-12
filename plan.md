# AdaptivePy — Implementation Plan (v1)

## 1. Overview

AdaptivePy is a modular Python package for performing adaptive sampling on molecular dynamics (MD) trajectories using clustering-based state space partitioning and policy-driven seed selection.

**Supports:**

- Feature-based trajectory inputs (required)
- Coordinate trajectories (optional, for seed extraction + PDB generation)
- Multiple clustering backends (sklearn-based)
- Multiple adaptive policies (extensible plugin system)
- Parallel policy evaluation in a single run
- CLI-driven execution via YAML configuration

---

## 2. Core Design Principles

### 2.1 Separation of Concerns

AdaptivePy is structured into independent layers:

| Layer | Responsibility |
|---|---|
| **IO** | Load features + trajectories |
| **Validation** | Consistency checks (traj/feature mapping) |
| **Clustering** | sklearn clustering |
| **State Statistics** | Cluster populations, assignments |
| **Policy** | Seed selection logic |
| **Seed Selection** | Frame-level selection within clusters |
| **Output** | Metadata + PDB writing |

### 2.2 Required Input Contract

**Features (required)**
```
features/
├── traj_0.npy   # (n_frames, n_features)
├── traj_1.npy
...
```

**Coordinates (optional)**
```
trajectories/
├── traj_0.xtc
├── traj_1.xtc
```

**Topology** (required if coordinates are used): `topology.pdb` / `topology.parm7`

> **Key rule:** Feature and trajectory filenames **must** match exactly — e.g. `traj_0.npy` ↔ `traj_0.xtc`.

---

## 3. High-Level Workflow

```
LOAD DATA
   ↓
VALIDATE MAPPING (features ↔ trajectories)
   ↓
CONCATENATE FEATURES (per-traj tracking preserved)
   ↓
CLUSTERING (sklearn)
   ↓
COMPUTE CLUSTER STATISTICS
   ↓
APPLY POLICY (can be multiple)
   ↓
SELECT SEEDS (frame-level selection)
   ↓
OPTIONAL PDB GENERATION
   ↓
WRITE OUTPUTS (metadata + seeds)
```

---

## 4. Data Model

### 4.1 FrameRecord

Every frame is tracked explicitly for full traceability:

```python
FrameRecord(
    traj_id: int,
    frame_id: int,
    features: np.ndarray,
    cluster_id: int
)
```

### 4.2 Internal Dataset Structure

```python
Dataset:
    frames: List[FrameRecord]
    feature_matrix: (N_total_frames, n_features)
    traj_index_map: dict
```

---

## 5. Clustering Module

### 5.1 Supported Methods (v1)

Implemented via sklearn:

- `KMeans`
- `MiniBatchKMeans`
- `AgglomerativeClustering` (optional extension)
- `RegularSpace` (custom wrapper if needed)

### 5.2 Interface

```python
class Clusterer(ABC):
    def fit(self, X): pass
    def predict(self, X): pass
```

### 5.3 Implementations

- `SklearnKMeansClusterer`
- `SklearnMiniBatchClusterer`
- `SklearnRegularSpaceClusterer`

### 5.4 Output

- `cluster_id` per frame
- `cluster_centers_`
- `cluster_model.pkl`

---

## 6. State Statistics Layer

Computed after clustering:

```python
cluster_stats = {
    cluster_id: {
        "population": int,
        "frames": List[FrameRecord]
    }
}
```

**Outputs:**

- Cluster populations
- Sorted cluster indices (ascending population)
- Frame assignments
- Per-cluster frame lists

---

## 7. Policy System

### 7.1 Base Class

```python
class Policy(ABC):
    name: str

    def select_clusters(self, cluster_stats, n_seeds):
        pass
```

### 7.2 Implemented Policies (v1)

#### Least Counts

Sort clusters by ascending population and select the top `n_seeds` clusters (one seed per cluster).

```python
sorted_clusters = sort_by_population(ascending=True)
selected = sorted_clusters[:n_seeds]
```

#### Random

Uniformly sample `n_seeds` cluster IDs at random.

### 7.3 Extensibility

New policies are auto-registered via:

```python
POLICY_REGISTRY = {}
```

```python
class MyPolicy(Policy):
    name = "my_policy"

    def select_clusters(...):
        ...
```

---

## 8. Seed Selection Module

Once clusters are chosen, a frame is selected from each.

### 8.1 YAML Configuration

```yaml
seed_selection:
    method: nearest_center | random_frame
```

### 8.2 Methods

| Method | Description |
|---|---|
| `nearest_center` *(default)* | Frame closest to cluster centroid |
| `random_frame` | Uniform random frame within cluster |

---

## 9. Multi-Policy Execution

### YAML Example

```yaml
policies:
  - least_counts
  - random
```

### Output Structure

```
results/
├── least_counts/
│   ├── seeds.csv
│   ├── metadata.csv
│   └── pdbs/
│
├── random/
│   ├── seeds.csv
│   ├── metadata.csv
│   └── pdbs/
│
└── combined_metadata.csv
```

---

## 10. Coordinate Handling (Optional)

If trajectories are provided, mdtraj is used to extract selected frames and save PDB files:

```python
traj.load_xtc(...)
traj[frame_id].save_pdb(...)
```

---

## 11. Output Specification

### 11.1 `seeds.csv`

```
seed_id,policy,traj_id,frame_id,cluster_id
0,least_counts,3,102,5
1,least_counts,1,88,2
```

### 11.2 Cluster Statistics

```
cluster_id,population
0,1234
1,532
```

### 11.3 Saved Artifacts

| File | Description |
|---|---|
| `cluster_model.pkl` | Serialised clustering model |
| `assignments.npy` | Per-frame cluster assignments |
| `run_config.yaml` | Exact copy of run configuration |
| `logs.txt` | Full run log |

---

## 12. CLI Design

**Primary command:**

```bash
adaptivepy run config.yaml
```

**Additional commands (v1.1):**

```bash
adaptivepy validate config.yaml
adaptivepy inspect clusters
adaptivepy list-policies
```

---

## 13. Project Structure

```
adaptivepy/
│
├── api.py
├── config/
│   └── schema.py
│
├── io/
│   ├── loader.py
│   └── trajectory.py
│
├── clustering/
│   ├── base.py
│   ├── sklearn_kmeans.py
│   ├── sklearn_minibatch.py
│   └── regular_space.py
│
├── policies/
│   ├── base.py
│   ├── least_counts.py
│   └── random.py
│
├── selection/
│   └── frame_selector.py
│
├── stats/
│   └── cluster_stats.py
│
├── output/
│   ├── writer.py
│   └── pdb_writer.py
│
├── cli/
│   └── run.py
│
└── utils/
    ├── logging.py
    └── io_utils.py
```

---

## 14. Reproducibility Requirements

Every run must save:

- Exact copy of `config.yaml`
- Random seeds
- Cluster model
- Frame assignments
- Full metadata

---

## 15. Performance Considerations

- Use `MiniBatchKMeans` for large datasets
- Avoid concatenating full feature copies unnecessarily
- Stream frame mapping where possible

---

## 16. v1 Scope Summary

### Must-Have

- sklearn clustering
- 2 policies (least counts + random)
- CLI + YAML config
- Feature input + optional trajectory input
- Metadata + PDB outputs
- Multi-policy support

### Nice-to-Have (include if straightforward)

- Cluster model saving
- Nearest-center seed selection
- Logging system

---

## 17. Future Extensions (Post-v1)

- MSM integration
- VAMP-based clustering
- Uncertainty-based policies
- Reinforcement-learning adaptive policies
- Streaming adaptive sampling
- Distributed HPC execution
