"""HfModelStore against a monkeypatched ``snapshot_download``: no network, no real cache."""

import logging
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import hf_thread_map
from huggingface_hub.utils._xet_progress_reporting import (
    XET_BYTES_BAR_FORMAT,
    XET_TRANSFER_BAR_FORMAT,
    XetDownloadProgressReporter,
    _finish_transfer_bar,  # pyright: ignore[reportPrivateUsage]
    _set_aggregate_rate_postfix,  # pyright: ignore[reportPrivateUsage]
    _update_transfer_bar,  # pyright: ignore[reportPrivateUsage]
)
from huggingface_hub.utils.tqdm import _create_progress_bar  # pyright: ignore[reportPrivateUsage]

from voxlektor.adapters.models import hf_cache
from voxlektor.adapters.models.hf_cache import HfModelStore
from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import ServiceUnavailable
from voxlektor.testing.fakes import FAKE_MODEL

INFO = ModelInfo(
    engine_id="x",
    title="X",
    repo="org/x",
    revision="rev1",
    size_bytes=100,
    license_name="MIT",
    license_url="https://example.org",
    commercial_use=True,
)


@dataclass
class _Hub:
    """Records calls; ``cached`` decides local lookups; ``fail`` is raised by downloads."""

    cached: bool = True
    fail: BaseException | None = None
    # A small file first, then the weights: the early total is only the small file's size.
    files: list[tuple[str, list[int]]] = field(
        default_factory=lambda: [("config.json", [10]), ("model.safetensors", [45, 45])]
    )
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    progress: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        tqdm_class: type[object] | None = None,
    ) -> str:
        self.calls.append(
            {"repo": repo_id, "revision": revision, "local": local_files_only, "bar": tqdm_class}
        )
        if local_files_only:
            if not self.cached:
                raise LocalEntryNotFoundError("not cached")
            return "/cache/snapshot"
        if self.fail is not None:
            raise self.fail
        assert tqdm_class is not None
        _snapshot_flow(tqdm_class, self.files)
        return "/cache/snapshot"


def _snapshot_flow(bar_class: type[object], files: list[tuple[str, list[int]]]) -> None:
    """What ``snapshot_download`` of huggingface_hub 2.4 does with ``tqdm_class``.

    Built from the library's own helpers (bar creation, the aggregate rate postfix that reads
    ``format_dict``, the transfer bar updates, ``hf_thread_map``), with its per-file
    ``_AggregatedTqdm`` inlined: each file adds its size to the totals when it starts, so the
    "Reconstructing" total is incomplete until the last file has started.
    """
    log_level = logging.INFO
    transfer = _create_progress_bar(  # pyright: ignore[reportPrivateUsage]
        cls=bar_class,
        log_level=log_level,
        name="huggingface_hub.snapshot_download.transfer",
        desc="Downloading bytes",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        bar_format=XET_TRANSFER_BAR_FORMAT,
    )
    rebuild = _create_progress_bar(  # pyright: ignore[reportPrivateUsage]
        cls=bar_class,
        log_level=log_level,
        name="huggingface_hub.snapshot_download",
        desc="Reconstructing (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        bar_format=XET_BYTES_BAR_FORMAT,
    )

    def download(file: tuple[str, list[int]]) -> None:
        _, chunks = file
        rebuild.total = (rebuild.total or 0) + sum(chunks)
        transfer.total = (transfer.total or 0) + sum(chunks)
        rebuild.refresh()
        for chunk in chunks:
            rebuild.update(chunk)
            _set_aggregate_rate_postfix(rebuild)  # pyright: ignore[reportPrivateUsage]
            _update_transfer_bar(transfer, chunk)  # pyright: ignore[reportPrivateUsage]
            _set_aggregate_rate_postfix(transfer)  # pyright: ignore[reportPrivateUsage]

    hf_thread_map(
        download, files, desc=f"Fetching {len(files)} files", max_workers=1, tqdm_class=bar_class
    )
    _finish_transfer_bar(transfer)  # pyright: ignore[reportPrivateUsage]
    transfer.set_description_str("Download complete")
    rebuild.set_description("Reconstruction complete")


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> _Hub:
    fake = _Hub()
    monkeypatch.setattr(hf_cache, "snapshot_download", fake.snapshot_download)
    return fake


def test_present_snapshot_is_looked_up_offline(hub: _Hub) -> None:
    assert HfModelStore().is_present(INFO) is True
    assert hub.calls == [{"repo": "org/x", "revision": "rev1", "local": True, "bar": None}]


def test_missing_snapshot_is_absent(hub: _Hub) -> None:
    hub.cached = False
    assert HfModelStore().is_present(INFO) is False


def test_engine_without_a_repo_is_always_present(hub: _Hub) -> None:
    store = HfModelStore()
    assert store.is_present(FAKE_MODEL) is True
    store.download(FAKE_MODEL, lambda done, total: None)
    assert hub.calls == []


def test_download_pins_repo_and_revision_and_reports_bytes(hub: _Hub) -> None:
    """The model's known size floors the growing total: no early 100% after the small file."""
    seen: list[tuple[int, int]] = []
    HfModelStore().download(INFO, lambda done, total: seen.append((done, total)))
    (call,) = hub.calls
    assert (call["repo"], call["revision"], call["local"]) == ("org/x", "rev1", False)
    assert seen == [(0, 100), (10, 100), (10, 100), (55, 100), (100, 100), (100, 100)]


def test_download_ends_at_100_percent_when_the_known_size_is_rounded_up(hub: _Hub) -> None:
    seen: list[tuple[int, int]] = []
    HfModelStore().download(
        replace(INFO, size_bytes=200), lambda done, total: seen.append((done, total))
    )
    assert seen[-2:] == [(100, 200), (100, 100)]


def test_download_larger_than_the_known_size_follows_the_library_total(hub: _Hub) -> None:
    seen: list[tuple[int, int]] = []
    HfModelStore().download(
        replace(INFO, size_bytes=50), lambda done, total: seen.append((done, total))
    )
    assert seen == [(0, 50), (10, 50), (10, 100), (55, 100), (100, 100), (100, 100)]


def test_bar_survives_the_libraries_per_file_xet_reporter() -> None:
    """``XetDownloadProgressReporter`` (hf_hub_download's Xet path) drives the bar class too."""
    seen: list[tuple[int, int]] = []
    tracker = hf_cache._Tracker(lambda done, total: seen.append((done, total)), 0)  # pyright: ignore[reportPrivateUsage]
    bar_class = hf_cache._bar_class(tracker)  # pyright: ignore[reportPrivateUsage]
    reporter = XetDownloadProgressReporter(
        reconstruction_desc="Reconstructing", log_level=logging.INFO, tqdm_class=bar_class
    )
    for done in (40, 100):
        report = SimpleNamespace(
            total_bytes=100,
            total_bytes_completed=done,
            total_bytes_completion_rate=1e6,
            total_transfer_bytes_completed=done,
            total_transfer_bytes_completion_rate=1e6,
        )
        reporter.update_progress(report)
    reporter.close()
    assert reporter.reconstruction_bar.n == 100
    assert seen[-1] == (100, 100)
    _set_aggregate_rate_postfix(reporter.transfer_bar)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "error",
    [ConnectionError("offline"), OSError("disk full"), RuntimeError("read timeout")],
    ids=["offline", "os", "transport"],
)
def test_download_failure_is_service_unavailable(hub: _Hub, error: BaseException) -> None:
    hub.fail = error
    with pytest.raises(ServiceUnavailable, match="model download failed: "):
        HfModelStore().download(INFO, lambda done, total: None)


def test_bar_without_a_tracker_counts_silently() -> None:
    bar = hf_cache._Bar(desc="Reconstructing", total=10)  # pyright: ignore[reportPrivateUsage]
    bar.update(5)
    bar.update(None)
    assert bar.n == 5
    assert bar.disable is True  # never draws on the terminal


def test_finish_without_any_report_says_nothing() -> None:
    seen: list[tuple[int, int]] = []
    hf_cache._Tracker(lambda done, total: seen.append((done, total)), 100).finish()  # pyright: ignore[reportPrivateUsage]
    assert seen == []
