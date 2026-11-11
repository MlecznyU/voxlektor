"""EventHub: fan-out of server events to subscribers.

Groundwork for protocol v1 ``subscribe`` (reserved; the MVP daemon answers ``unsupported`` and
nothing subscribes yet). Loop thread only.
"""

from collections.abc import Callable, Mapping

type Subscriber = Callable[[str, Mapping[str, object]], None]


class EventHub:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Add ``subscriber``; return a function that removes it (safe to call twice)."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    def publish(self, event: str, data: Mapping[str, object]) -> None:
        for subscriber in list(self._subscribers):
            subscriber(event, data)
