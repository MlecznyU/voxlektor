"""Module entry point: ``python -m voxlektor`` (also how the daemon is spawned)."""

import sys

from voxlektor.entrypoints.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
