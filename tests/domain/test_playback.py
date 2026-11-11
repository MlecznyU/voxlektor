from collections import deque
from copy import deepcopy

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from voxlektor.domain.playback import PlaybackQueue


def _queue(history: int = 20) -> PlaybackQueue[int]:
    return PlaybackQueue[int](history=history)


def test_default_history_cap_is_20() -> None:
    q = PlaybackQueue[int]()
    for i in range(30):
        q.push(i)
        q.start_next()
        q.finish_current()
    assert len(q.history) == 20


def test_fifo_order_and_properties() -> None:
    q = _queue()
    assert q.start_next() is None
    assert q.current is None
    for i in (1, 2, 3):
        q.push(i)
    assert q.queued == 3
    assert q.start_next() == 1
    assert q.current == 1
    assert q.queued == 2
    q.finish_current()
    assert q.current is None
    assert q.history == (1,)
    assert q.start_next() == 2


def test_finish_without_current_is_noop() -> None:
    q = _queue()
    q.finish_current()
    assert q.history == ()


def test_skip_drops_current_without_history_and_returns_next() -> None:
    q = _queue()
    for i in (1, 2):
        q.push(i)
    q.start_next()
    assert q.skip() == 2
    assert q.current == 2
    assert q.history == ()
    assert q.skip() is None
    assert q.current is None


def test_back_requeues_previous_and_current_at_front() -> None:
    q = _queue()
    for i in (1, 2, 3):
        q.push(i)
    q.start_next()
    q.finish_current()
    q.start_next()
    assert q.back() == [1, 2]
    assert q.current is None
    assert q.history == ()
    assert q.queued == 3
    assert q.start_next() == 1
    q.finish_current()
    assert q.start_next() == 2


def test_back_without_history_restarts_current() -> None:
    q = _queue()
    q.push(1)
    q.push(2)
    q.start_next()
    assert q.back() == [1]
    assert q.current is None
    assert q.queued == 2
    assert q.start_next() == 1
    q.finish_current()
    assert q.start_next() == 2


def test_back_with_history_but_no_current_requeues_previous() -> None:
    q = _queue()
    q.push(1)
    q.start_next()
    q.finish_current()
    assert q.back() == [1]
    assert q.history == ()
    assert q.start_next() == 1


def test_back_with_nothing_is_noop() -> None:
    q = _queue()
    assert q.back() == []
    assert q.current is None
    assert q.queued == 0


def test_history_is_capped() -> None:
    q = _queue(history=20)
    for i in range(30):
        q.push(i)
        q.start_next()
        q.finish_current()
    assert len(q.history) == 20
    assert q.history[0] == 10
    assert q.history[-1] == 29


def test_clear_returns_current_and_queued() -> None:
    q = _queue()
    for i in (1, 2, 3):
        q.push(i)
    q.start_next()
    assert q.clear() == [1, 2, 3]
    assert q.current is None
    assert q.queued == 0
    assert q.clear() == []


def test_clear_keeps_history() -> None:
    q = _queue()
    q.push(1)
    q.start_next()
    q.finish_current()
    q.push(2)
    assert q.clear() == [2]
    assert q.history == (1,)


def test_pause_and_resume() -> None:
    q = _queue()
    assert not q.paused
    q.pause()
    assert q.paused
    q.resume()
    assert not q.paused


class PlaybackMachine(RuleBasedStateMachine):
    """Runs the real queue next to an explicit model that tracks every item by identity."""

    CAP = 3

    def __init__(self) -> None:
        super().__init__()
        self.q = PlaybackQueue[int](history=self.CAP)
        self.next_id = 0
        self.queue: deque[int] = deque()
        self.history: list[int] = []
        self.current: int | None = None
        self.pushed: set[int] = set()
        self.dropped: set[int] = set()  # skip, clear or history eviction
        self.paused = False

    @rule()
    def push(self) -> None:
        self.q.push(self.next_id)
        self.queue.append(self.next_id)
        self.pushed.add(self.next_id)
        self.next_id += 1

    @rule()
    def start_next(self) -> None:
        got = self.q.start_next()
        if self.current is None and self.queue:
            self.current = self.queue.popleft()
        assert got == (self.current if self.current is not None else None)

    @rule()
    def finish(self) -> None:
        self.q.finish_current()
        if self.current is not None:
            self.history.append(self.current)
            self.current = None
            while len(self.history) > self.CAP:
                self.dropped.add(self.history.pop(0))

    @rule()
    def skip(self) -> None:
        got = self.q.skip()
        if self.current is not None:
            self.dropped.add(self.current)
        self.current = self.queue.popleft() if self.queue else None
        assert got == self.current

    @rule()
    def back(self) -> None:
        got = self.q.back()
        expected: list[int] = []
        if self.history:
            expected.append(self.history.pop())
        if self.current is not None:
            expected.append(self.current)
            self.current = None
        self.queue.extendleft(reversed(expected))
        assert got == expected

    @rule(flag=st.booleans())
    def pause_resume(self, flag: bool) -> None:
        self.paused = flag
        if flag:
            self.q.pause()
        else:
            self.q.resume()

    @rule()
    def clear(self) -> None:
        got = self.q.clear()
        expected = [*([self.current] if self.current is not None else []), *self.queue]
        self.dropped.update(expected)
        self.current = None
        self.queue.clear()
        assert got == expected

    def real_live(self) -> list[int]:
        """The REAL queue's [current, *queued], read via clear() on a throwaway copy."""
        return deepcopy(self.q).clear()

    @invariant()
    def matches_model(self) -> None:
        expected = [*([self.current] if self.current is not None else []), *self.queue]
        assert self.real_live() == expected
        assert self.q.current == self.current
        assert self.q.queued == len(self.queue)
        assert self.q.history == tuple(self.history)
        assert self.q.paused == self.paused

    @invariant()
    def history_is_capped(self) -> None:
        assert len(self.q.history) <= self.CAP

    @invariant()
    def no_item_is_in_two_places(self) -> None:
        live = self.real_live()
        everything = [*live, *self.q.history]
        assert len(everything) == len(set(everything))
        if self.q.current is not None:
            assert live.count(self.q.current) == 1

    @invariant()
    def no_item_is_lost(self) -> None:
        held = {*self.real_live(), *self.q.history}
        assert self.pushed - held == self.dropped


TestPlaybackMachine = PlaybackMachine.TestCase
