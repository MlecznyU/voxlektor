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
