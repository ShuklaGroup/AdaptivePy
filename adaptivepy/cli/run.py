"""Command-line interface for AdaptivePy."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from adaptivepy import __version__
from adaptivepy.api import run_adaptive_sampling, validate_config
from adaptivepy.policies import list_policies


@click.group()
@click.version_option(version=__version__, prog_name="adaptivepy")
def main() -> None:
    """AdaptivePy: clustering-based adaptive sampling for MD trajectories."""


@main.command("run")
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def run_cmd(config: Path) -> None:
    """Run adaptive sampling from a YAML configuration file.

    Parameters
    ----------
    config : Path
        Path to the YAML configuration file.
    """
    try:
        run_adaptive_sampling(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command("validate")
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def validate_cmd(config: Path) -> None:
    """Validate configuration and input data without running clustering.

    Parameters
    ----------
    config : Path
        Path to the YAML configuration file.
    """
    try:
        validate_config(config)
        click.echo(f"Configuration valid: {config}")
    except Exception as exc:
        click.echo(f"Validation failed: {exc}", err=True)
        sys.exit(1)


@main.command("list-policies")
def list_policies_cmd() -> None:
    """List all registered adaptive sampling policies."""
    policies = list_policies()
    if not policies:
        click.echo("No policies registered.")
        return
    for name in policies:
        click.echo(name)


if __name__ == "__main__":
    main()
