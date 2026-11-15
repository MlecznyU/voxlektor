"""EventHub: fan-out of server events to subscribers.

``CoreLoop`` publishes every ``Emit`` effect of the Session here, in effect order; bootstrap
creates one hub per daemon and hands bounded queues (``subscribe_bounded``) to the socket
server, one per protocol v1 ``subscribe`` connection. Loop thread only.

Bounded subscribers never lose an event silently: when ``limit`` events are waiting in a queue,
the next event is replaced by the overflow marker ``None`` and that subscriber is removed. The
queue has room for exactly ``limit`` events plus the marker, so ``queue.full()`` means "overflowed"
(the server checks it while a slow client blocks its writes, then closes that connection).
"""

import asyncio
from collections.abc import Callable, Mapping

type Subscriber = Callable[[str, Mapping[str, object]], None]
type _Queue = asyncio.Queue[tuple[str, Mapping[str, object]] | None]


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

    def subscribe_bounded(
        self, limit: int
    ) -> tuple[asyncio.Queue[tuple[str, Mapping[str, object]] | None], Callable[[], None]]:
        """A queue that receives every event, at most ``limit`` waiting; ``None`` = overflow.

        Returns the queue and the function that unsubscribes it (safe to call twice).
        """
        queue: _Queue = asyncio.Queue(limit + 1)

        def push(event: str, data: Mapping[str, object]) -> None:
            if queue.qsize() < limit:
                queue.put_nowait((event, data))
                return
            unsubscribe()
            queue.put_nowait(None)

        unsubscribe = self.subscribe(push)
        return queue, unsubscribe

    def publish(self, event: str, data: Mapping[str, object]) -> None:
        for subscriber in list(self._subscribers):
            subscriber(event, data)


class BoundedSubscription:
    """One stream subscriber (the socket server's ``Subscription``): a bounded hub queue, taken
    when built, and ``snapshot``, the status at that moment (called in the same loop step)."""

    def __init__(
        self, hub: EventHub, limit: int, snapshot: Callable[[], Mapping[str, object]]
    ) -> None:
        self._queue, self._unsubscribe = hub.subscribe_bounded(limit)
        self._snapshot = snapshot

    def snapshot(self) -> Mapping[str, object]:
        return self._snapshot()

    def queue(self) -> _Queue:
        return self._queue

    def close(self) -> None:
        self._unsubscribe()
