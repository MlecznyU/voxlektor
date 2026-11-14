"""Chunk size limits shared by ``split_text`` and the reading policy (not a frozen module)."""

MAX_CHARS = 300
MERGE_BELOW = 60
FIRST_MAX_CHARS = 120  # the first chunk of a reading: shorter, so the first audio comes sooner
