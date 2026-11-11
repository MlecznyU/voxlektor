import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from voxlektor.domain.keepwarm import (
    Action,
    Busy,
    CancelWake,
    Event,
    FlushPending,
    Idle,
    KeepWarm,
    Life,
    Loaded,
    LoadFailed,
    NeedEngine,
    ScheduleWake,
    SetTimeout,
    StartLoad,
    StartUnload,
    Unloaded,
    UnloadRequested,
    Wake,
    transition,
)


def _s(
    life: Life,
    timeout_s: int = 30,
    idle: float | None = None,
    unload_after_load: bool = False,
    reload: bool = False,
) -> KeepWarm:
    # The deadline of an idle WARM state is always idle + timeout (None for timeout 0).
    deadline = None if idle is None or timeout_s == 0 else idle + timeout_s
    return KeepWarm(life, timeout_s, deadline, unload_after_load, reload, idle)


C, L, W, U = Life.COLD, Life.LOADING, Life.WARM, Life.UNLOADING

# (id, state, event, expected state, expected actions)
RULES: list[tuple[str, KeepWarm, Event, KeepWarm, list[Action]]] = [
    # COLD
    ("cold-need", _s(C), NeedEngine(), _s(L), [StartLoad()]),
    ("cold-unload-requested", _s(C), UnloadRequested(), _s(C), []),
    ("cold-loaded-ignored", _s(C), Loaded(), _s(C), []),
    ("cold-load-failed-ignored", _s(C), LoadFailed("x"), _s(C), []),
    ("cold-idle-ignored", _s(C), Idle(1.0), _s(C), []),
    ("cold-busy-ignored", _s(C), Busy(), _s(C), []),
    ("cold-wake-ignored", _s(C), Wake(1.0), _s(C), []),
    ("cold-unloaded-ignored", _s(C), Unloaded(), _s(C), []),
    ("cold-set-timeout", _s(C), SetTimeout(5), _s(C, 5), []),
    # LOADING
    ("loading-need", _s(L), NeedEngine(), _s(L), []),
    ("loading-loaded", _s(L), Loaded(), _s(W), [FlushPending()]),
    (
        "loading-loaded-unload-after",
        _s(L, unload_after_load=True),
        Loaded(),
        _s(U),
        [FlushPending(), StartUnload()],
    ),
    ("loading-unload-requested", _s(L), UnloadRequested(), _s(L, unload_after_load=True), []),
    (
        "loading-unload-requested-twice",
        _s(L, unload_after_load=True),
        UnloadRequested(),
        _s(L, unload_after_load=True),
        [],
    ),
    ("loading-load-failed", _s(L), LoadFailed("boom"), _s(C), []),
    (
        "loading-load-failed-clears-flag",
        _s(L, unload_after_load=True),
        LoadFailed("boom"),
        _s(C),
        [],
    ),
    ("loading-idle-ignored", _s(L), Idle(1.0), _s(L), []),
    ("loading-busy-ignored", _s(L), Busy(), _s(L), []),
    ("loading-wake-ignored", _s(L), Wake(1.0), _s(L), []),
    ("loading-unloaded-ignored", _s(L), Unloaded(), _s(L), []),
    ("loading-set-timeout", _s(L), SetTimeout(5), _s(L, 5), []),
    # WARM
    ("warm-need-ignored", _s(W), NeedEngine(), _s(W), []),
    ("warm-loaded-ignored", _s(W), Loaded(), _s(W), []),
    ("warm-load-failed-ignored", _s(W), LoadFailed("x"), _s(W), []),
    ("warm-unloaded-ignored", _s(W), Unloaded(), _s(W), []),
    (
        "warm-idle",
        _s(W, 30),
        Idle(100.0),
        _s(W, 30, idle=100.0),
        [ScheduleWake(130.0)],
    ),
    ("warm-idle-timeout-zero", _s(W, 0), Idle(100.0), _s(W, 0, idle=100.0), []),
    ("warm-busy", _s(W, 30, idle=100.0), Busy(), _s(W, 30), [CancelWake()]),
    ("warm-busy-timeout-zero-idle", _s(W, 0, idle=100.0), Busy(), _s(W, 0), []),
    ("warm-busy-already-busy", _s(W), Busy(), _s(W), []),
    (
        "warm-wake-at-deadline",
        _s(W, 30, idle=100.0),
        Wake(130.0),
        _s(U, 30),
        [StartUnload()],
    ),
    (
        "warm-wake-after-deadline",
        _s(W, 30, idle=100.0),
        Wake(140.0),
        _s(U, 30),
        [StartUnload()],
    ),
    (
        "warm-wake-stale-early",
        _s(W, 30, idle=100.0),
        Wake(129.0),
        _s(W, 30, idle=100.0),
        [],
    ),
    ("warm-wake-no-deadline", _s(W), Wake(500.0), _s(W), []),
    ("warm-unload-requested", _s(W), UnloadRequested(), _s(U), [StartUnload()]),
    (
        "warm-unload-requested-cancels-wake",
        _s(W, 30, idle=100.0),
        UnloadRequested(),
        _s(U, 30),
        [CancelWake(), StartUnload()],
    ),
    (
        "warm-unload-requested-timeout-zero-idle",
        _s(W, 0, idle=100.0),
        UnloadRequested(),
        _s(U, 0),
        [StartUnload()],
    ),
    # WARM + SetTimeout (idle started at 100)
    ("warm-set-timeout-busy", _s(W), SetTimeout(5), _s(W, 5), []),
    (
        "warm-set-timeout-0-to-600",
        _s(W, 0, idle=100.0),
        SetTimeout(600),
        _s(W, 600, idle=100.0),
        [ScheduleWake(700.0)],
    ),
    (
        "warm-set-timeout-600-to-0",
        _s(W, 600, idle=100.0),
        SetTimeout(0),
        _s(W, 0, idle=100.0),
        [CancelWake()],
    ),
    (
        "warm-set-timeout-0-to-0",
        _s(W, 0, idle=100.0),
        SetTimeout(0),
        _s(W, 0, idle=100.0),
        [],
    ),
    (
        "warm-set-timeout-600-to-300",
        _s(W, 600, idle=100.0),
        SetTimeout(300),
        _s(W, 300, idle=100.0),
        [ScheduleWake(400.0)],
    ),
    (
        "warm-set-timeout-shorter-than-elapsed",
        _s(W, 600, idle=100.0),
        SetTimeout(10),
        _s(W, 10, idle=100.0),
        [ScheduleWake(110.0)],
    ),
    (
        "warm-set-timeout-negative",
        _s(W, 30, idle=100.0),
        SetTimeout(-5),
        _s(W, 0, idle=100.0),
        [CancelWake()],
    ),
    # UNLOADING
    ("unloading-unloaded", _s(U), Unloaded(), _s(C), []),
    ("unloading-need", _s(U), NeedEngine(), _s(U, reload=True), []),
    ("unloading-need-twice", _s(U, reload=True), NeedEngine(), _s(U, reload=True), []),
    ("unloading-unloaded-reload", _s(U, reload=True), Unloaded(), _s(L), [StartLoad()]),
    ("unloading-unload-requested-ignored", _s(U), UnloadRequested(), _s(U), []),
    (
        "unloading-unload-requested-after-need",
        _s(U, reload=True),
        UnloadRequested(),
        _s(U, unload_after_load=True, reload=True),
        [],
    ),
    (
        "unloading-unloaded-reload-keeps-unload-after-load",
        _s(U, unload_after_load=True, reload=True),
        Unloaded(),
        _s(L, unload_after_load=True),
        [StartLoad()],
    ),
    ("unloading-loaded-ignored", _s(U), Loaded(), _s(U), []),
    ("unloading-load-failed-ignored", _s(U), LoadFailed("x"), _s(U), []),
    ("unloading-idle-ignored", _s(U), Idle(1.0), _s(U), []),
    ("unloading-busy-ignored", _s(U), Busy(), _s(U), []),
    ("unloading-wake-ignored", _s(U), Wake(1.0), _s(U), []),
    ("unloading-set-timeout", _s(U), SetTimeout(5), _s(U, 5), []),
]


@pytest.mark.parametrize(
    ("state", "event", "expected_state", "expected_actions"),
    [pytest.param(s, e, es, ea, id=i) for i, s, e, es, ea in RULES],
)
def test_transition_table(
    state: KeepWarm, event: Event, expected_state: KeepWarm, expected_actions: list[Action]
) -> None:
    new_state, actions = transition(state, event)
    assert new_state == expected_state
    assert actions == expected_actions


def test_load_failure_then_need_loads_again() -> None:
    s, _ = transition(_s(C), NeedEngine())
    s, _ = transition(s, LoadFailed("x"))
    s, actions = transition(s, NeedEngine())
    assert (s.life, actions) == (L, [StartLoad()])


def test_full_cycle_with_reload_during_unload() -> None:
    s = _s(C)
    for event in (NeedEngine(), Loaded(), Idle(10.0), Wake(40.0), NeedEngine()):
        s, _ = transition(s, event)
    s, actions = transition(s, Unloaded())
    assert (s.life, actions) == (L, [StartLoad()])
    assert s.unload_after_load is False
    assert s.reload_after_unload is False


def test_explicit_unload_survives_a_reload_during_unloading() -> None:
    s = _s(C)
    for event in (NeedEngine(), Loaded(), UnloadRequested(), NeedEngine(), UnloadRequested()):
        s, _ = transition(s, event)
    s, actions = transition(s, Unloaded())
    assert (s.life, actions) == (L, [StartLoad()])
    s, actions = transition(s, Loaded())
    assert (s.life, actions) == (U, [FlushPending(), StartUnload()])
    s, actions = transition(s, Unloaded())
    assert (s.life, actions) == (C, [])


_EVENTS = st.one_of(
    st.just(NeedEngine()),
    st.just(Loaded()),
    st.builds(LoadFailed, st.just("err")),
    st.builds(Idle, st.floats(0, 1000)),
    st.just(Busy()),
    st.just(UnloadRequested()),
    st.just(Unloaded()),
    st.builds(Wake, st.floats(0, 1000)),
    st.builds(SetTimeout, st.integers(0, 100)),
)


def _assert_timer_shape(s: KeepWarm) -> None:
    if s.life is not Life.WARM:
        assert s.deadline is None
        assert s.idle_since is None
    if s.idle_since is None or s.timeout_s == 0:
        assert s.deadline is None
    else:
        assert s.deadline == s.idle_since + s.timeout_s


class KeepWarmMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.state = KeepWarm(Life.COLD, 30, None, False)
        self.loading = False  # a StartLoad is outstanding
        self.owed_unload = False  # an UnloadRequested that has not yet led to StartUnload

    @rule(event=_EVENTS)
    def step(self, event: Event) -> None:
        before = self.state
        after, actions = transition(before, event)

        starts = [a for a in actions if isinstance(a, StartLoad)]
        assert len(starts) <= 1
        if starts:
            assert not self.loading
            self.loading = True
        if isinstance(event, (Loaded, LoadFailed)) and before.life is Life.LOADING:
            self.loading = False

        self._track_requested_unload(before, event, actions, after)

        if after.timeout_s == 0:
            assert not any(isinstance(a, ScheduleWake) for a in actions)
        if isinstance(event, Wake) and (before.deadline is None or event.now < before.deadline):
            assert after == before
            assert actions == []
        for a in actions:
            if isinstance(a, ScheduleWake):
                assert after.deadline == a.at
        _assert_timer_shape(after)
        self.state = after

    def _track_requested_unload(
        self, before: KeepWarm, event: Event, actions: list[Action], after: KeepWarm
    ) -> None:
        if isinstance(event, UnloadRequested) and (
            before.life in (Life.LOADING, Life.WARM)
            or (before.life is Life.UNLOADING and before.reload_after_unload)
        ):
            self.owed_unload = True
        if any(isinstance(a, StartUnload) for a in actions):
            assert after.life is Life.UNLOADING
            self.owed_unload = False
        if after.life is Life.COLD:
            self.owed_unload = False

    @invariant()
    def loading_flag_matches_life(self) -> None:
        assert self.loading == (self.state.life is Life.LOADING)

    @invariant()
    def requested_unload_is_never_lost(self) -> None:
        if self.owed_unload:
            assert self.state.life is not Life.WARM


TestKeepWarmMachine = KeepWarmMachine.TestCase
