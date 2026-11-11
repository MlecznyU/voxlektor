# pyright: basic
"""Generate ``docs/protocol/v1.schema.json`` from the protocol contract (never edit it by hand)."""

import json
from pathlib import Path

import msgspec

from voxlektor.contracts.protocol import Event, Reply, Request

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "protocol" / "v1.schema.json"


def render_schema() -> str:
    schemas, components = msgspec.json.schema_components([Request, Reply, Event])
    document = {
        "$defs": components,
        "request": schemas[0],
        "reply": schemas[1],
        "event": schemas[2],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_schema(), encoding="utf-8")


if __name__ == "__main__":
    main()
