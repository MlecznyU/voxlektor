"""Engines are discovered through ``voxlektor.engines`` entry points, never imported.

Their model descriptions come from a second, light group, ``voxlektor.engine_models``, whose
targets import no ML framework, so setup can list engines without loading any of them.
"""

from importlib.metadata import entry_points

from voxlektor.bootstrap import environment
from voxlektor.contracts.engine import Engine, ModelInfo
from voxlektor.domain.errors import DomainError

_GROUP = "voxlektor.engines"
_MODELS_GROUP = "voxlektor.engine_models"


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


def engine_models(include_hidden: bool = False) -> list[ModelInfo]:
    """The registered engines' model descriptions, sorted by engine id."""
    models: list[ModelInfo] = []
    for point in entry_points(group=_MODELS_GROUP):
        info = point.load()
        if not isinstance(info, ModelInfo):
            raise TypeError(f"entry point {point.name!r} is not a ModelInfo")
        if include_hidden or not info.hidden:
            models.append(info)
    return sorted(models, key=lambda info: info.engine_id)
