import os

import pytest

from voxlektor.bootstrap import environment
from voxlektor.bootstrap.registry import load_engine
from voxlektor.domain.errors import DomainError
from voxlektor.testing.fakes import FakeEngine

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


def test_load_engine_sets_the_hub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HF_VAR, raising=False)
    load_engine("fake")
    assert os.environ[HF_VAR] == "1"


def test_prepare_does_not_override_an_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_VAR, "0")
    environment.prepare()
    assert os.environ[HF_VAR] == "0"
