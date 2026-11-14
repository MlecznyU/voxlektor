from typing import Any

XET_TRANSFER_BAR_FORMAT: str
XET_BYTES_BAR_FORMAT: str

def _set_aggregate_rate_postfix(bar: Any) -> None: ...
def _update_transfer_bar(bar: Any, inc: int) -> None: ...
def _finish_transfer_bar(bar: Any) -> None: ...

class XetDownloadProgressReporter:
    reconstruction_bar: Any
    transfer_bar: Any
    def __init__(
        self,
        *,
        reconstruction_desc: str,
        transfer_desc: str = ...,
        total: int | None = ...,
        log_level: int,
        name: str | None = ...,
        tqdm_class: type[object] | None = ...,
        external_reconstruction_bar: Any | None = ...,
        position: int = ...,
    ) -> None: ...
    def update_progress(
        self, group_report: Any, _item_reports: dict[str, Any] | None = ...
    ) -> None: ...
    def close(self) -> None: ...
