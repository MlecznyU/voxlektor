"""Module entry point: ``python -m voxlektor``."""

# The CLI module arrives in Task 10; drop the pyright ignore then.
from voxlektor.entrypoints.cli.app import main  # pyright: ignore

if __name__ == "__main__":
    main()
