"""PortAudio re-initialisation, shared by the sounddevice output and the cue player.

PortAudio fixes the device list and the default output device when it is initialised (at
``import sounddevice``). The daemon lives for hours, so without a re-initialisation a default
device that changed later (headphones plugged in, AirPods connected, a display with speakers)
would be ignored, and a device that vanished would keep failing. ``refresh_devices`` terminates
and re-initialises PortAudio so that the next stream opens on the current default device.

``Pa_Terminate`` closes every open stream of the process, so the refresh runs only when no
stream is open: every adapter that opens streams registers as a ``StreamOwner`` and is asked to
release its stream first (an owner whose stream still plays refuses, and the refresh waits for
the next opportunity). Owners are asked OUTSIDE ``_lock``: closing a stream may wait for its
finished callback on the PortAudio thread, which may call ``mark_stale``.

Failure safety: ``_initialized`` tracks whether PortAudio is initialised. If terminate fails,
the library is still initialised and the refresh stays due; if initialise fails after a
successful terminate, the library is left uninitialised, the refresh stays due and the next one
only initialises (it never terminates an uninitialised library). ``refresh_devices`` raises the
``PortAudioError`` so the caller can report it (the output turns it into a playback failure).

Threads: everything runs on the loop thread except ``mark_stale``, which the PortAudio thread
may call when a playback fails; it is lock-free (one plain bool store, atomic under the GIL), so
it can never block on a refresh in progress.
"""

import threading
import weakref
from collections.abc import Callable
from typing import Protocol

import sounddevice as sd


class StreamOwner(Protocol):
    def release_idle_stream(self) -> bool:
        """Close the stream if it no longer plays; True when this owner has no open stream."""
        ...


_lock = threading.Lock()  # guards _owners; serialises refreshes
_owners: weakref.WeakSet[StreamOwner] = weakref.WeakSet()
_due = False  # written lock-free by mark_stale (any thread)
_initialized = True  # sounddevice initialises PortAudio when it is imported


def register(owner: StreamOwner) -> None:
    with _lock:
        _owners.add(owner)


def mark_stale() -> None:
    """A playback failed (perhaps its device vanished): refresh before the next stream opens.

    Lock-free on purpose (see the module docstring): a plain bool store.
    """
    global _due  # noqa: PLW0603  # one process-wide flag, like PortAudio's own state
    _due = True


def refresh_devices(*, idle: bool) -> bool:
    """Re-initialise PortAudio when ``idle``, after a failure or while it is uninitialised, if no
    stream is open. Returns True when PortAudio was re-initialised; raises ``PortAudioError``
    (and stays due) when terminate or initialise failed.
    """
    global _due  # noqa: PLW0603
    if not (idle or _due or not _initialized):
        return False
    with _lock:
        owners = list(_owners)
    released = [owner.release_idle_stream() for owner in owners]  # ask every owner, unlocked
    if not all(released):
        return False  # a stream still plays; try again at the next opportunity
    with _lock:
        _due = False  # a failure noticed meanwhile sets it again and is honoured next time
        try:
            _reinitialize()
        except BaseException:
            _due = True
            raise
    return True


def _private(name: str) -> Callable[[], None]:
    # sounddevice has no public API to re-scan the devices; ``_terminate``/``_initialize`` are
    # its own (private) wrappers of Pa_Terminate/Pa_Initialize, kept in this one place.
    function: Callable[[], None] = getattr(sd, name)
    return function


def _reinitialize() -> None:
    global _initialized  # noqa: PLW0603
    if _initialized:
        _private("_terminate")()  # raises with the library still initialised
        _initialized = False
    _private("_initialize")()  # raises with the library left uninitialised
    _initialized = True
