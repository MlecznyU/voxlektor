from collections.abc import Mapping

from voxlektor.runtime.events import EventHub


def test_publish_reaches_every_subscriber_until_it_unsubscribes() -> None:
    hub = EventHub()
    first: list[tuple[str, Mapping[str, object]]] = []
    second: list[tuple[str, Mapping[str, object]]] = []
    unsubscribe = hub.subscribe(lambda event, data: first.append((event, data)))
    hub.subscribe(lambda event, data: second.append((event, data)))
    hub.publish("state", {"state": "WARM"})
    unsubscribe()
    unsubscribe()  # idempotent
    hub.publish("state", {"state": "COLD"})
    assert first == [("state", {"state": "WARM"})]
    assert second == [("state", {"state": "WARM"}), ("state", {"state": "COLD"})]


def test_publish_without_subscribers_is_a_no_op() -> None:
    EventHub().publish("state", {})


def test_subscribe_bounded_queues_events_in_order_until_unsubscribed() -> None:
    hub = EventHub()
    queue, unsubscribe = hub.subscribe_bounded(5)
    hub.publish("floor", {"owner": "a"})
    hub.publish("drained", {"owner": "a"})
    unsubscribe()
    unsubscribe()  # idempotent
    hub.publish("state", {"state": "cold"})
    assert [queue.get_nowait(), queue.get_nowait()] == [
        ("floor", {"owner": "a"}),
        ("drained", {"owner": "a"}),
    ]
    assert queue.empty()


def test_subscribe_bounded_overflow_ends_with_none_and_unsubscribes() -> None:
    hub = EventHub()
    slow, _ = hub.subscribe_bounded(3)
    other, _ = hub.subscribe_bounded(1000)
    for n in range(5):
        hub.publish("tag_started", {"n": n})
    assert slow.full()  # full means the overflow marker is in
    seen = [slow.get_nowait() for _ in range(slow.qsize())]
    assert seen == [*(("tag_started", {"n": n}) for n in range(3)), None]
    hub.publish("tag_started", {"n": 5})
    assert slow.empty()  # no longer subscribed
    assert other.qsize() == 6  # the other subscriber is intact
