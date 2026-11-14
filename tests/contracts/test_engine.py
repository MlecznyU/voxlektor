"""The light model description every engine publishes next to its (heavy) implementation."""

import dataclasses

import pytest

from voxlektor.contracts.engine import ModelInfo


def _info(**overrides: object) -> ModelInfo:
    values: dict[str, object] = {
        "engine_id": "x",
        "title": "X",
        "repo": "org/x",
        "revision": "abc",
        "size_bytes": 10,
        "license_name": "MIT",
        "license_url": "https://example.org/x",
        "commercial_use": True,
    }
    return ModelInfo(**(values | overrides))  # pyright: ignore[reportArgumentType]


def test_model_info_is_not_hidden_by_default() -> None:
    assert _info().hidden is False
    assert _info(hidden=True).hidden is True


def test_model_info_is_frozen() -> None:
    info = _info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.repo = "other"  # type: ignore[misc]
