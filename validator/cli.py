"""Command-line interface for validator table generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_REPORT_NAME
from .discovery import discover_collections
from .loader import load_collection
from .reporting import render_collection_report, render_combined_report
from .validators import build_default_validators


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate validator markdown tables for repo run folders.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Repo root, collection directory, or single run directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Combined output path or single-collection output path.",
    )
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Only include run directories whose name contains this text.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    base_path = Path(args.workdir).resolve()
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return 1

    collections = discover_collections(base_path)
    if not collections:
        print(f"Error: no run collections found under {base_path}")
        return 1

    validators = build_default_validators()
    loaded_collections = [
        load_collection(
            collection_path=collection,
            run_filter=args.run_filter,
            verbose=not args.quiet,
        )
        for collection in collections
    ]
    loaded_collections = [
        collection
        for collection in loaded_collections
        if any(collection.checkpoints_by_reg.values())
    ]
    if not loaded_collections:
        print("Error: no checkpoints found after loading collections")
        return 1

    output_path = Path(args.output).resolve() if args.output else None
    if output_path is not None:
        report = render_output(loaded_collections, validators)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote combined report to {output_path}")
        return 0

    for collection in loaded_collections:
        report = render_collection_report(collection, validators)
        destination = collection.path / DEFAULT_REPORT_NAME
        destination.write_text(report, encoding="utf-8")
        print(f"Wrote report to {destination}")
    return 0


def render_output(collections, validators) -> str:
    """Render either a single collection or a combined report."""
    if len(collections) == 1:
        return render_collection_report(collections[0], validators)
    return render_combined_report(collections, validators)
