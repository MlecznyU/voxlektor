"""QuestionaryPrompter with questionary monkeypatched: no terminal is needed."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
import questionary
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from voxlektor.adapters.terminal.menus import QuestionaryPrompter
from voxlektor.app.ports import Option
from voxlektor.app.setup import SetupCancelled


@dataclass
class _Question:
    """Stands in for ``questionary.Question``: answers ``answer`` (or raises it)."""

    answer: object
    application: Any = field(default_factory=lambda: SimpleNamespace(key_bindings=None))

    def unsafe_ask(self) -> object:
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@dataclass
class _Calls:
    made: list[tuple[str, tuple[object, ...], dict[str, object]]] = field(
        default_factory=list[tuple[str, tuple[object, ...], dict[str, object]]]
    )
    questions: list[_Question] = field(default_factory=list[_Question])


def _patch(monkeypatch: pytest.MonkeyPatch, kind: str, answer: object) -> _Calls:
    calls = _Calls()

    def make(*args: object, **kwargs: object) -> _Question:
        calls.made.append((kind, args, kwargs))
        question = _Question(answer)
        calls.questions.append(question)
        return question

    monkeypatch.setattr(questionary, kind, make)
    return calls


def test_choose_shows_labels_and_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, "select", "b")
    options = [Option("a", "Alpha"), Option("b", "Beta")]
    assert QuestionaryPrompter().choose("Pick one", options, "a") == "b"
    ((_, args, kwargs),) = calls.made
    assert args == ("Pick one",)
    assert kwargs["default"] == "a"
    choices = cast("list[questionary.Choice]", kwargs["choices"])
    assert [(c.title, c.value) for c in choices] == [("Alpha", "a"), ("Beta", "b")]


def test_number_parses_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, "text", "1.25")
    assert QuestionaryPrompter().number("Rate", 0.5, 3.0, 1.0) == 1.25
    ((_, _, kwargs),) = calls.made
    assert kwargs["default"] == "1"
    check = kwargs["validate"]
    assert callable(check)
    assert check("2") is True
    assert check("9") == "enter a number from 0.5 to 3"
    assert check("fast") == "enter a number from 0.5 to 3"


def test_confirm_passes_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, "confirm", False)
    assert QuestionaryPrompter().confirm("Keep?", default=True) is False
    assert calls.made == [("confirm", ("Keep?",), {"default": True})]


@pytest.mark.parametrize("kind", ["select", "text", "confirm"])
def test_no_answer_is_setup_cancelled(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    _patch(monkeypatch, kind, None)
    prompter = QuestionaryPrompter()
    ask = {
        "select": lambda: prompter.choose("q", [Option("a", "A")], "a"),
        "text": lambda: prompter.number("q", 0, 1, 0),
        "confirm": lambda: prompter.confirm("q", default=True),
    }[kind]
    with pytest.raises(SetupCancelled):
        ask()


def test_ctrl_c_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, "confirm", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        QuestionaryPrompter().confirm("q", default=True)


class _App:
    def __init__(self) -> None:
        self.results: list[object] = []

    def exit(self, result: object = "unset") -> None:
        self.results.append(result)


@pytest.mark.parametrize("has_bindings", [False, True])
def test_escape_ends_the_question_without_an_answer(
    monkeypatch: pytest.MonkeyPatch, has_bindings: bool
) -> None:
    existing = KeyBindings()
    question = _Question(True, SimpleNamespace(key_bindings=existing if has_bindings else None))

    def make(*args: object, **kwargs: object) -> _Question:
        return question

    monkeypatch.setattr(questionary, "confirm", make)
    QuestionaryPrompter().confirm("q", default=True)
    bindings = question.application.key_bindings.bindings
    (escape,) = [b for b in bindings if b.keys == (Keys.Escape,)]
    app = _App()
    escape.handler(SimpleNamespace(app=app))
    assert app.results == [None]


def test_say_prints_a_line(capsys: pytest.CaptureFixture[str]) -> None:
    QuestionaryPrompter().say("hello")
    assert capsys.readouterr().out == "hello\n"
