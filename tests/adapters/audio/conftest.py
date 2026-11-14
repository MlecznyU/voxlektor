"""Audio adapter test kit: PortAudio is never really re-initialised in device-free tests."""

import weakref
from collections.abc import Callable

import pytest
import sounddevice as sd

from voxlektor.adapters.audio import portaudio


class Reinits:
    """Stands in for sounddevice's private ``_terminate``/``_initialize``.

    ``count`` is the number of completed re-initialisations; ``check`` runs at each terminate
    (e.g. "no stream is open"). ``fail_terminate`` / ``fail_initialize`` make the next N calls
    raise ``PortAudioError``. Like PortAudio, terminating an uninitialised library is an error.
    """

    def __init__(self) -> None:
        self.count = 0
        self.initialized = True
        self.terminates = 0
        self.initializes = 0
        self.fail_terminate = 0
        self.fail_initialize = 0
        self.check: Callable[[], None] | None = None

    def terminate(self) -> None:
        self.terminates += 1
        assert self.initialized, "terminated an uninitialised PortAudio"
        if self.check is not None:
            self.check()
        if self.fail_terminate:
            self.fail_terminate -= 1
            raise sd.PortAudioError("Error terminating PortAudio")
        self.initialized = False

    def initialize(self) -> None:
        self.initializes += 1
        assert not self.initialized, "initialised twice without terminating"
        if self.fail_initialize:
            self.fail_initialize -= 1
            raise sd.PortAudioError("Error initializing PortAudio")
        self.initialized = True
        self.count += 1


@pytest.fixture(autouse=True)
def reinits(monkeypatch: pytest.MonkeyPatch) -> Reinits:
    recorder = Reinits()
    monkeypatch.setattr(sd, "_terminate", recorder.terminate)
    monkeypatch.setattr(sd, "_initialize", recorder.initialize)
    monkeypatch.setattr(portaudio, "_owners", weakref.WeakSet[portaudio.StreamOwner]())
    monkeypatch.setattr(portaudio, "_due", False)
    monkeypatch.setattr(portaudio, "_initialized", True)
    return recorder
