from voxlektor.domain.floor import Floor

A1 = ("a", "1")
A2 = ("a", "2")
B1 = ("b", "1")
C1 = ("c", "1")


def test_first_group_gets_floor() -> None:
    f = Floor(8)
    assert f.holder is None
    assert f.hand_over(0) is None
    f.add(A1, 0)
    assert f.hand_over(0) == A1
    assert f.holder == A1
    assert f.hand_over(1) == A1


def test_fifo_across_owners() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.add(B1, 1)
    f.add(A2, 2)
    assert f.hand_over(2) == A1
    assert f.groups_of("a") == (A1, A2)
    f.remove(A1)
    assert f.holder is None
    assert f.hand_over(3) == B1
    f.remove(B1)
    assert f.hand_over(4) == A2


def test_paused_waiting_owner_passed_over() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.add(B1, 1)
    f.add(C1, 2)
    f.hand_over(2)
    f.pause_owner("b", 2)
    assert f.paused("b")
    f.remove(A1)
    assert f.hand_over(3) == C1
    f.resume_owner("b", 3)
    assert not f.paused("b")
    f.resume_owner("b", 3)
    f.remove(C1)
    assert f.hand_over(4) == B1


def test_paused_holder_keeps_floor_without_waiters() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    assert f.grace_deadline() is None
    f.pause_owner("a", 1)
    assert f.grace_deadline() is None
    assert f.on_time(1000) is False
    assert f.hand_over(1000) == A1
    f.add(B1, 5)
    f.pause_owner("b", 5)
    assert f.grace_deadline() is None


def test_unpaused_holder_has_no_deadline() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    assert f.grace_deadline() is None
    assert f.on_time(100) is False


def test_grace_starts_at_first_waiter() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.pause_owner("a", 0)
    f.pause_owner("a", 3)
    f.add(B1, 5)
    f.add(C1, 9)
    assert f.grace_deadline() == 13


def test_grace_starts_at_pause_when_waiter_first() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 0)
    f.pause_owner("a", 3)
    assert f.grace_deadline() == 11


def test_yield_then_resume_is_next() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("a", 2)
    assert f.on_time(9) is False
    assert f.on_time(10) is True
    assert f.grace_deadline() is None
    assert f.play_order() == (B1,)
    assert f.hand_over(10) == B1
    f.resume_owner("a", 10)
    assert f.play_order() == (B1, A1)
    assert f.position(A1) == 1
    f.remove(B1)
    assert f.hand_over(20) == A1


def test_yielded_holder_with_nobody_eligible_stays() -> None:
    f = Floor(0)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("a", 1)
    assert f.on_time(1) is True
    f.remove(B1)
    assert f.hand_over(2) == A1
    assert f.play_order() == ()


def test_zero_grace_yields_at_once() -> None:
    f = Floor(0)
    f.add(A1, 0)
    f.hand_over(0)
    f.pause_owner("a", 1)
    f.add(B1, 5)
    assert f.grace_deadline() == 5
    assert f.on_time(5) is True


def test_age_inheritance() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 5)
    f.add(C1, 10, age=1)
    assert f.play_order() == (A1, C1, B1)
    f.remove(A1)
    assert f.hand_over(10) == C1


def test_preempt_returns_interrupted_to_front() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.add(C1, 2)
    assert f.preempt(C1, 3) == A1
    assert f.holder == C1
    assert f.play_order() == (C1, A1, B1)
    assert f.position(A1) == 1


def test_preempt_without_holder_and_of_holder() -> None:
    f = Floor(8)
    f.add(A1, 0)
    assert f.preempt(A1, 0) is None
    assert f.holder == A1
    assert f.preempt(A1, 1) is None
    assert f.holder == A1


def test_preempt_paused_interrupted_holder_is_passed_over() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("a", 2)
    f.on_time(10)
    f.preempt(B1, 11)
    assert f.play_order() == (B1,)


def test_hold_blocks_hand_over() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("a", 1)
    assert f.grace_deadline() == 9
    f.hold()
    assert f.held
    assert f.on_time(1000) is False
    assert f.grace_deadline() is None
    f.remove(A1)
    assert f.hand_over(1000) is None
    f.release(1001)
    assert not f.held
    assert f.hand_over(1001) == B1


def test_grace_from_late_eligibility() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("b", 2)
    f.pause_owner("a", 3)
    assert f.grace_deadline() is None
    f.resume_owner("b", 100)  # b becomes eligible only now
    assert f.grace_deadline() == 108
    assert f.on_time(107) is False
    assert f.on_time(108) is True
    assert f.hand_over(108) == B1
    f.pause_owner("b", 110)  # the new holder pauses; the yielded a is still paused
    assert f.grace_deadline() is None
    f.resume_owner("a", 200)  # the yielded holder becomes eligible only now
    assert f.grace_deadline() == 208


def test_grace_restarts_after_hold_release() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.hand_over(0)
    f.add(B1, 1)
    f.pause_owner("a", 2)
    f.hold()
    f.release(100)
    assert f.grace_deadline() == 108
    f.release(200)  # not held: nothing restarts
    assert f.grace_deadline() == 108


def test_play_order_and_position() -> None:
    f = Floor(8)
    f.add(A1, 0)
    f.add(B1, 1)
    f.add(C1, 2)
    f.add(A2, 3)
    f.hand_over(3)
    f.pause_owner("b", 3)
    assert f.play_order() == (A1, C1, A2)
    assert f.position(A1) is None
    assert f.position(B1) is None
    assert f.position(C1) == 1
    assert f.position(A2) == 2
    assert f.position(("z", "z")) is None
