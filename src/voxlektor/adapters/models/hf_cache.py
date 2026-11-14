"""ModelStore on the Hugging Face cache: is the pinned snapshot there, and download it if not.

The download runs in the setup process (the daemon then loads offline). Progress comes from
``snapshot_download``'s ``tqdm_class`` hook. The bar class subclasses ``huggingface_hub``'s own
``tqdm`` (always disabled, so it never draws), which gives it every attribute the library uses
(``format_dict``, locks, ``set_postfix_str``, ...); only ``update``/``refresh`` are overridden
to report the "Reconstructing" bar (the bytes written for every file) as ``progress(done, total)``.

That bar's total is incomplete while files start: each file adds its size when it begins, so a
small first file would read as an early 100%. The model's known size (``ModelInfo.size_bytes``)
is therefore a floor for the total, and a finished download reports ``(done, done)`` once.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import tqdm as hf_tqdm

from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import ServiceUnavailable

type _Progress = Callable[[int, int], None]

_REPORTED_BAR = "Reconstructing"


@dataclass(slots=True)
class _Tracker:
    """One download's progress: ``floor`` is the model's known size in bytes."""

    progress: _Progress
    floor: int
    done: int = 0

    def seen(self, done: int, total: int) -> None:
        self.done = done
        total = max(total, self.floor)
        if total > 0:
            self.progress(done, total)

    def finish(self) -> None:
        if self.done > 0:
            self.progress(self.done, self.done)


class _Bar(hf_tqdm):
    """A silent huggingface_hub ``tqdm`` that reports the "Reconstructing" bar to a tracker."""

    tracker: ClassVar[_Tracker | None] = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        desc = kwargs.get("desc")
        kwargs["disable"] = True  # never draw; a disabled tqdm keeps no description itself
        super().__init__(*args, **kwargs)
        self.desc = desc if isinstance(desc, str) else ""

    def update(self, n: float | None = 1) -> bool | None:
        self.n += n or 0  # a disabled tqdm does not count
        self._report()
        return None

    def refresh(self, nolock: bool = False, lock_args: object = None) -> None:
        self._report()

    def _report(self) -> None:
        tracker = type(self).tracker
        if tracker is not None and self.desc.startswith(_REPORTED_BAR):
            tracker.seen(int(self.n), int(self.total or 0))


def _bar_class(tracker: _Tracker) -> type[_Bar]:
    class _ReportingBar(_Bar):
        pass

    _ReportingBar.tracker = tracker
    return _ReportingBar


class HfModelStore:
    def is_present(self, info: ModelInfo) -> bool:
        """The pinned snapshot is complete in the local cache (no network is used)."""
        if not info.repo:
            return True
        try:
            snapshot_download(info.repo, revision=info.revision, local_files_only=True)
        except LocalEntryNotFoundError:
            return False
        return True

    def download(self, info: ModelInfo, progress: Callable[[int, int], None]) -> None:
        """Fetch the pinned snapshot (resuming a partial one); failures are ServiceUnavailable."""
        if not info.repo:
            return
        tracker = _Tracker(progress, info.size_bytes)
        try:
            snapshot_download(info.repo, revision=info.revision, tqdm_class=_bar_class(tracker))
        # The Hub raises OSError subclasses (HTTP, offline, cache) but also bare transport
        # errors of its HTTP client; any of them means "the download did not finish".
        except Exception as exc:
            raise ServiceUnavailable(f"model download failed: {exc}") from None
        tracker.finish()
