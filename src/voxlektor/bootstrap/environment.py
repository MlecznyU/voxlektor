"""Process-wide settings that must be in place before any engine is imported."""

import os

_HF_HUB_SWITCH = "HF_HUB_DISABLE_IMPLICIT_TOKEN"


def prepare() -> None:
    """Never let huggingface_hub send a (possibly stale) implicit token.

    Assigned, not defaulted: an inherited ``=0`` must not switch the implicit token back on.
    """
    os.environ[_HF_HUB_SWITCH] = "1"
