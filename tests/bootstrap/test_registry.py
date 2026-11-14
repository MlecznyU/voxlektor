import os
import subprocess
import sys

import pytest

from voxlektor.bootstrap import environment, registry
from voxlektor.bootstrap.registry import engine_models, load_engine
from voxlektor.domain.errors import DomainError
from voxlektor.testing.fakes import FAKE_MODEL, FakeEngine

HF_VAR = "HF_HUB_DISABLE_IMPLICIT_TOKEN"


def test_load_engine_returns_the_fake_engine() -> None:
    engine = load_engine("fake")
    assert isinstance(engine, FakeEngine)
    assert engine.spec.id == "fake"


def test_load_engine_returns_a_new_instance_each_time() -> None:
    assert load_engine("fake") is not load_engine("fake")


def test_unknown_engine_lists_the_available_ids() -> None:
    with pytest.raises(DomainError) as info:
        load_engine("nope")
    assert info.value.code == "unknown_engine"
    assert "'nope'" in info.value.message
    assert "available: fake, omnivoice" in info.value.message


def test_engine_models_lists_the_visible_engines() -> None:
    assert [info.engine_id for info in engine_models()] == ["omnivoice"]


def test_engine_models_with_hidden_adds_the_fake_engine() -> None:
    models = engine_models(include_hidden=True)
    assert [info.engine_id for info in models] == ["fake", "omnivoice"]
    assert models[0] is FAKE_MODEL


def test_engine_models_never_imports_an_engine() -> None:
    code = (
        "import sys; from voxlektor.bootstrap.registry import engine_models; engine_models(); "
        "heavy = {'mlx', 'mlx_audio', 'voxlektor.adapters.engines.omnivoice_mlx'}; "
        "bad = heavy & set(sys.modules); assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_engine_models_rejects_a_foreign_object(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Point:
        name = "odd"

        def load(self) -> object:
            return "not a model"

    def fake_entry_points(group: str) -> list[_Point]:
        return [_Point()]

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)
    with pytest.raises(TypeError, match="'odd'"):
        engine_models()


def test_load_engine_sets_the_hub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HF_VAR, raising=False)
    load_engine("fake")
    assert os.environ[HF_VAR] == "1"


def test_prepare_overrides_an_inherited_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """S2: an inherited ``0`` must not switch the implicit Hub token back on."""
    monkeypatch.setenv(HF_VAR, "0")
    environment.prepare()
    assert os.environ[HF_VAR] == "1"
