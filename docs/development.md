# Development

Guide for contributing to AdaptivePy and building documentation locally.

## Setup

```bash
git clone https://github.com/hnadeem2/AdaptivePy.git
cd AdaptivePy
pip install -e ".[dev,docs]"
```

## Run tests

```bash
pytest tests/ -q
```

## Build documentation locally

Serve with live reload:

```bash
mkdocs serve
```

Open [http://127.0.0.1:8000/AdaptivePy/](http://127.0.0.1:8000/AdaptivePy/) in your browser.

Build a static site (same check used in CI):

```bash
mkdocs build --strict
```

Output is written to `site/`.

## Documentation structure

| Path | Purpose |
|------|---------|
| `docs/*.md` | User guides (edit these for workflow docs) |
| `docs/reference/*.md` | API reference via mkdocstrings (pulls from code docstrings) |
| `mkdocs.yml` | Site theme, navigation, and plugin config |

When you update docstrings in `adaptivepy/`, the API reference pages update
automatically on the next docs build.

## Project layout

```text
adaptivepy/
├── api.py              # Main workflow
├── config/             # YAML schema
├── io/                 # Feature and trajectory loading
├── clustering/         # Clustering backends
├── policies/           # Adaptive policies
├── selection/          # Frame-level seed selection
├── stats/              # Cluster statistics
├── output/             # Writers (CSV, PDB, model)
└── cli/                # Command-line interface
```

## Adding a new policy

1. Create a module in `adaptivepy/policies/`
2. Subclass `Policy` and use `@register_policy`
3. Import the module in `adaptivepy/policies/__init__.py`
4. Add tests and document in [Policies](policies.md)

## CI workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/docs.yml` | Push to `main` (docs/code changes) | Build and deploy GitHub Pages |
| `.github/workflows/publish.yml` | GitHub Release published | Publish to PyPI |

## Release checklist

1. Update version in `pyproject.toml` and `adaptivepy/__init__.py`
2. Run `pytest tests/ -q` and `mkdocs build --strict`
3. Merge to `main` (docs deploy automatically)
4. Create a GitHub Release to trigger PyPI publish
