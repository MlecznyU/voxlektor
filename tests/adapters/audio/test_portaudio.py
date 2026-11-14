"""PortAudio re-initialisation: only when wanted (idle or after a failure) and no stream is open."""

import threading

import pytest
import sounddevice as sd

from tests.adapters.audio.conftest import Reinits
from voxlektor.adapters.audio import portaudio


class Owner:
    """A ``StreamOwner`` whose stream is playing until told otherwise."""

    def __init__(self, *, playing: bool) -> None:
        self.playing = playing
        self.asked = 0

    def release_idle_stream(self) -> bool:
        self.asked += 1
        return not self.playing


def test_nothing_happens_when_not_idle_and_nothing_failed(reinits: Reinits) -> None:
    owner = Owner(playing=False)
    portaudio.register(owner)
    assert portaudio.refresh_devices(idle=False) is False
    assert owner.asked == 0
    assert reinits.count == 0


def test_idle_refresh_re_initialises_once(reinits: Reinits) -> None:
    assert portaudio.refresh_devices(idle=True) is True
    assert reinits.count == 1


def test_a_failure_makes_the_next_refresh_due_once(reinits: Reinits) -> None:
    portaudio.mark_stale()
    assert portaudio.refresh_devices(idle=False) is True
    assert portaudio.refresh_devices(idle=False) is False  # done: no longer due
    assert reinits.count == 1


def test_a_playing_stream_postpones_the_refresh_and_every_owner_is_asked(
    reinits: Reinits,
) -> None:
    busy, idle = Owner(playing=True), Owner(playing=False)
    portaudio.register(busy)
    portaudio.register(idle)
    portaudio.mark_stale()
    assert portaudio.refresh_devices(idle=True) is False
    assert (busy.asked, idle.asked) == (1, 1)
    assert reinits.count == 0
    busy.playing = False
    assert portaudio.refresh_devices(idle=False) is True  # still due after the postponement
    assert reinits.count == 1


def test_initialise_failing_once_is_retried_without_a_second_terminate(
    reinits: Reinits,
) -> None:
    """R1: after terminate succeeded and initialise failed, the next refresh only initialises."""
    reinits.fail_initialize = 1
    with pytest.raises(sd.PortAudioError, match="initializing"):
        portaudio.refresh_devices(idle=True)
    assert reinits.initialized is False
    assert portaudio.refresh_devices(idle=False) is True  # still due (and uninitialised)
    assert (reinits.terminates, reinits.initializes, reinits.count) == (1, 2, 1)
    assert reinits.initialized
    assert portaudio.refresh_devices(idle=False) is False  # recovered: nothing due any more


def test_terminate_failing_keeps_it_due_and_initialised(reinits: Reinits) -> None:
    reinits.fail_terminate = 1
    with pytest.raises(sd.PortAudioError, match="terminating"):
        portaudio.refresh_devices(idle=True)
    assert reinits.initialized  # never initialised on top of a live library
    assert reinits.initializes == 0
    assert portaudio.refresh_devices(idle=False) is True  # due: the next one retries in full
    assert (reinits.terminates, reinits.initializes, reinits.count) == (2, 1, 1)


def _mark_stale_from_another_thread() -> None:
    """What the PortAudio thread does; it must finish while a refresh is in progress."""
    worker = threading.Thread(target=portaudio.mark_stale)
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "mark_stale blocked on a refresh in progress"


def test_mark_stale_never_blocks_while_owners_release_their_streams(reinits: Reinits) -> None:
    """R2: closing a stream may wait for a finished callback that calls mark_stale."""

    class WaitsForFinishedCallback:
        def release_idle_stream(self) -> bool:
            _mark_stale_from_another_thread()
            return True

    owner = WaitsForFinishedCallback()
    portaudio.register(owner)
    assert portaudio.refresh_devices(idle=True) is True
    # The failure was marked before the re-initialisation, which therefore already covered it.
    assert portaudio.refresh_devices(idle=False) is False


def test_mark_stale_never_blocks_during_the_re_initialisation(reinits: Reinits) -> None:
    reinits.check = _mark_stale_from_another_thread
    assert portaudio.refresh_devices(idle=True) is True
    reinits.check = None
    assert portaudio.refresh_devices(idle=False) is True  # marked during the re-init: due again


def test_owners_are_held_weakly(reinits: Reinits) -> None:
    owner = Owner(playing=True)
    portaudio.register(owner)
    assert portaudio.refresh_devices(idle=True) is False
    del owner  # a garbage owner never blocks a refresh
    assert portaudio.refresh_devices(idle=True) is True
