"""Floor scheduler: which group of which owner plays now (pure, time arrives as arguments)."""

from __future__ import annotations

type GroupKey = tuple[str, str]
"""(owner, group)."""


class Floor:
    """The floor goes to the oldest waiting group whose owner is not paused.

    A paused holder keeps the floor while nobody eligible waits; with an eligible waiter it
    yields after ``grace_s`` measured from the later of (pause, first eligible waiter arriving).
    A yielded holder is an ordinary waiting group at its age. ``hold`` is a daemon-wide
    suspension of every hand-over (no grace runs).

    A waiter *arrives* (becomes eligible) when it is added, when its owner resumes, when it is
    interrupted by a preempt, and when the hold is released; the grace counts from the latest
    of these, never from an earlier time it could not have played.
    """

    def __init__(self, grace_s: float) -> None:
        self._grace_s = grace_s
        self._age: dict[GroupKey, float] = {}
        self._arrival: dict[GroupKey, float] = {}  # when each group last became eligible
        self._paused_at: dict[str, float] = {}
        self._holder: GroupKey | None = None
        self._yielded = False
        self._held = False

    @property
    def holder(self) -> GroupKey | None:
        return self._holder

    @property
    def held(self) -> bool:
        return self._held

    def add(self, key: GroupKey, now: float, *, age: float | None = None) -> None:
        self._age[key] = now if age is None else age
        self._arrival[key] = now

    def remove(self, key: GroupKey) -> None:
        if key == self._holder:
            self._holder = None
            self._yielded = False
        self._age.pop(key, None)
        self._arrival.pop(key, None)

    def groups_of(self, owner: str) -> tuple[GroupKey, ...]:
        return tuple(k for k in self._by_age(self._age) if k[0] == owner)

    def pause_owner(self, owner: str, now: float) -> None:
        self._paused_at.setdefault(owner, now)

    def resume_owner(self, owner: str, now: float) -> None:
        if self._paused_at.pop(owner, None) is None:
            return
        for key in self._arrival:
            if key[0] == owner:
                self._arrival[key] = now

    def paused(self, owner: str) -> bool:
        return owner in self._paused_at

    def hold(self) -> None:
        self._held = True

    def release(self, now: float) -> None:
        if not self._held:
            return
        self._held = False
        for key in self._arrival:
            self._arrival[key] = now

    def hand_over(self, now: float) -> GroupKey | None:
        del now
        if self._held or (self._holder is not None and not self._yielded):
            return self._holder
        waiting = self._eligible_waiting()
        if waiting:
            self._holder = waiting[0]
            self._yielded = False
        return self._holder

    def preempt(self, key: GroupKey, now: float) -> GroupKey | None:
        if key == self._holder:
            self._yielded = False
            return None
        interrupted = self._holder
        if interrupted is not None:
            self._arrival[interrupted] = now
        self._holder = key
        self._yielded = False
        return interrupted

    def grace_deadline(self) -> float | None:
        holder = self._holder
        if holder is None or self._yielded or self._held:
            return None
        paused_at = self._paused_at.get(holder[0])
        if paused_at is None:
            return None
        waiting = self._eligible_waiting()
        if not waiting:
            return None
        first_arrival = min(self._arrival[k] for k in waiting)
        return max(paused_at, first_arrival) + self._grace_s

    def on_time(self, now: float) -> bool:
        deadline = self.grace_deadline()
        if deadline is None or now < deadline:
            return False
        self._yielded = True
        return True

    def play_order(self) -> tuple[GroupKey, ...]:
        head = (self._holder,) if self._holder is not None and not self._yielded else ()
        return head + tuple(self._eligible_waiting())

    def position(self, key: GroupKey) -> int | None:
        waiting = self._eligible_waiting()
        if key not in waiting:
            return None
        return waiting.index(key) + 1

    def _eligible_waiting(self) -> list[GroupKey]:
        skip = None if self._yielded else self._holder
        return [k for k in self._by_age(self._age) if k != skip and k[0] not in self._paused_at]

    @staticmethod
    def _by_age(ages: dict[GroupKey, float]) -> list[GroupKey]:
        return sorted(ages, key=ages.__getitem__)
