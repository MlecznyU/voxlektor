"""Prompter on ``questionary``: arrow-key menus, a validated number field, yes/no.

Imported only inside ``vxl setup`` (questionary and prompt_toolkit are never on the CLI control
path). Ctrl+C propagates as ``KeyboardInterrupt`` (exit 130); Esc, or any question that ends
without an answer (questionary returns ``None``), raises ``SetupCancelled`` (exit 1).
"""

from collections.abc import Callable
from typing import Any

import questionary
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings

from voxlektor.app.ports import Option
from voxlektor.app.setup import SetupCancelled


def _cancel_on_escape(question: questionary.Question) -> questionary.Question:
    """Make Esc end the question without an answer (questionary has no binding for it)."""
    escape = KeyBindings()

    @escape.add("escape", eager=True)
    def _cancel(event: KeyPressEvent) -> None:  # pyright: ignore[reportUnusedFunction]
        event.app.exit(result=None)

    app = question.application
    app.key_bindings = merge_key_bindings([app.key_bindings or KeyBindings(), escape])
    return question


def _answer(question: questionary.Question) -> Any:  # noqa: ANN401  # questionary's own type
    answer = _cancel_on_escape(question).unsafe_ask()
    if answer is None:
        raise SetupCancelled()
    return answer


def _number_check(low: float, high: float) -> Callable[[str], bool | str]:
    """A questionary validator: True, or the message to show under the field."""

    def check(text: str) -> bool | str:
        try:
            value = float(text)
        except ValueError:
            return f"enter a number from {low:g} to {high:g}"
        return low <= value <= high or f"enter a number from {low:g} to {high:g}"

    return check


class QuestionaryPrompter:
    def choose(self, question: str, options: list[Option], default: str) -> str:
        choices = [questionary.Choice(title=option.label, value=option.value) for option in options]
        return str(_answer(questionary.select(question, choices=choices, default=default)))

    def number(self, question: str, low: float, high: float, default: float) -> float:
        text = _answer(
            questionary.text(question, default=f"{default:g}", validate=_number_check(low, high))
        )
        return float(text)

    def confirm(self, question: str, default: bool) -> bool:
        return bool(_answer(questionary.confirm(question, default=default)))

    def say(self, line: str) -> None:
        print(line, flush=True)
