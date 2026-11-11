"""Process-wide settings that must be in place before any engine is imported."""

import os


def prepare() -> None:
    """Never let huggingface_hub send a (possibly stale) implicit token."""
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
