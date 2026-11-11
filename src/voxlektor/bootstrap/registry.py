"""Engines are discovered through ``voxlektor.engines`` entry points, never imported."""

from importlib.metadata import entry_points

from voxlektor.bootstrap import environment
from voxlektor.contracts.engine import Engine
from voxlektor.domain.errors import DomainError

_GROUP = "voxlektor.engines"


def load_engine(engine_id: str) -> Engine:
    """Import the engine registered as ``engine_id`` and return a new, unloaded instance."""
    environment.prepare()
    found = {point.name: point for point in entry_points(group=_GROUP)}
    point = found.get(engine_id)
    if point is None:
        available = ", ".join(sorted(found))
        raise DomainError("unknown_engine", f"unknown engine {engine_id!r}; available: {available}")
    engine: Engine = point.load()()
    return engine
