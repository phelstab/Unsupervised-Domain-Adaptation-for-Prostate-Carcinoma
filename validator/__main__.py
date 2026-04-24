"""Module entrypoint for ``python -m validator_table``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
