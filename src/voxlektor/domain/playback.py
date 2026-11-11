"""Playback queue: what is waiting, what is playing, what was played."""

from collections import deque


class PlaybackQueue[A]:
    """FIFO queue with one ``current`` item and a bounded history.

    Invariant: an item lives in exactly one of ``queued``, ``current`` or ``history``
    unless it was dropped by ``skip``/``clear`` or evicted from a full history.
    """

    def __init__(self, history: int = 20) -> None:
        self._cap = history
        self._queue: deque[A] = deque()
        self._history: deque[A] = deque()
        self._current: A | None = None
        self._paused = False

    @property
    def current(self) -> A | None:
        return self._current

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def history(self) -> tuple[A, ...]:
        return tuple(self._history)

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def push(self, item: A) -> None:
        self._queue.append(item)

    def start_next(self) -> A | None:
        """Make the next queued item current; return it (None when nothing is queued).

        An item that is already current stays current.
        """
        if self._current is None and self._queue:
            self._current = self._queue.popleft()
        return self._current

    def finish_current(self) -> None:
        """Move the current item to history (oldest entries beyond the cap are dropped)."""
        if self._current is None:
            return
        self._history.append(self._current)
        self._current = None
        while len(self._history) > self._cap:
            self._history.popleft()

    def skip(self) -> A | None:
        """Drop the current item without recording it; start and return the next one."""
        self._current = None
        return self.start_next()

    def back(self) -> list[A]:
        """Re-queue ``[previous, current]`` at the front and return them.

        Without a current item only the previous one is re-queued; without history only the
        current one (a restart); with neither nothing changes. Re-queued items leave history
        and ``current`` becomes None; the caller picks them up with ``start_next``.
        """
        items: list[A] = []
        if self._history:
            items.append(self._history.pop())
        if self._current is not None:
            items.append(self._current)
            self._current = None
        self._queue.extendleft(reversed(items))
        return items

    def clear(self) -> list[A]:
        """Drop the current and all queued items (history stays); return them in order."""
        dropped: list[A] = [] if self._current is None else [self._current]
        dropped.extend(self._queue)
        self._current = None
        self._queue.clear()
        return dropped
