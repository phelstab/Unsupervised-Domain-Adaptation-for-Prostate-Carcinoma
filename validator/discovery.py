"""Filesystem discovery helpers for run collections."""

from __future__ import annotations

from pathlib import Path


def is_run_directory(path: Path) -> bool:
    """Return ``True`` when the path looks like a training run."""
    return path.is_dir() and (path / "results.json").is_file()


def has_run_directories(path: Path) -> bool:
    """Return ``True`` when a directory directly contains run folders."""
    if not path.is_dir():
        return False
    return any(is_run_directory(child) for child in path.iterdir())


def discover_collections(base_path: Path) -> list[Path]:
    """Resolve a repo root, collection directory, or run directory."""
    if is_run_directory(base_path):
        return [base_path]
    if has_run_directories(base_path):
        return [base_path]

    collections = [
        child
        for child in sorted(base_path.iterdir())
        if child.is_dir() and has_run_directories(child)
    ]
    return collections


def discover_run_directories(collection_path: Path) -> list[Path]:
    """Return run directories that belong to a collection."""
    if is_run_directory(collection_path):
        return [collection_path]

    runs = [
        child
        for child in sorted(collection_path.iterdir())
        if is_run_directory(child)
    ]
    return runs
