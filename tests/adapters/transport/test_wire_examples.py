"""Every documented example and golden reply must match what the real code path produces.

Real path: wire request -> decode_request -> to_command -> Session (via the Simulator) ->
to_wire_reply -> encode_line. ``subscribe`` fails in ``to_command`` and goes through error_reply.
"""

import json
from pathlib import Path

import msgspec

from voxlektor.adapters.transport.mapping import error_reply, to_command, to_wire_reply
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts import protocol as wire
from voxlektor.contracts.codec import ProtocolError, decode_request, encode_line
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "contracts" / "protocol_v1"
DOC = ROOT / "docs" / "protocol.md"
COMMANDS = {
    "speak",
    "pause",
    "resume",
    "skip",
    "back",
    "stop_reading",
    "status",
    "idle",
    "unload",
    "shutdown",
    "subscribe",
}


def _sim() -> Simulator:
    clock = FakeClock()
    refs = [VoiceRef(n, Path(f"/v/{n}.wav"), n) for n in ("v0-default", "m1-calm")]
    settings = Settings(engine=EngineId("fake"))
    session = Session(settings, VoiceCatalog(InMemoryVoiceRepository(refs)), clock)
    return Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(auto_complete=True), clock)


def _real_reply(sim: Simulator, line: bytes) -> wire.Reply:
    try:
        req = decode_request(line)
        return to_wire_reply(req.id, sim.handle(to_command(req)))
    except ProtocolError as err:
        return error_reply(err)


def _lines(path: Path) -> list[bytes]:
    return [ln for ln in path.read_bytes().splitlines() if ln]


def _golden_replies() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for line in _lines(GOLDEN / "replies.jsonl"):
        rep_id = json.loads(line)["id"]
        if rep_id is not None:
            out[rep_id] = line
    return out


def _same_shape(real: object, doc: object, where: str) -> None:
    """Same keys, same value types (null matches anything), recursively for objects."""
    if real is None or doc is None:
        return
    assert type(real) is type(doc), where
    if isinstance(real, dict):
        real_obj = msgspec.convert(real, dict[str, object])
        doc_obj = msgspec.convert(doc, dict[str, object])
        assert real_obj.keys() == doc_obj.keys(), where
        for key, value in real_obj.items():
            _same_shape(value, doc_obj[key], f"{where}.{key}")


def _compare(real: wire.Reply, documented: bytes, where: str) -> None:
    """Exact bytes for deterministic replies; shape only for ``status``."""
    doc = json.loads(documented)
    if doc.get("data", {}).get("state") is None:
        assert encode_line(real) == documented + b"\n", where
    else:
        real_json = json.loads(encode_line(real))
        _same_shape(real_json, doc, where)
        assert real_json["data"]["caps"] == doc["data"]["caps"], f"{where}: caps"


def test_golden_requests_get_the_documented_real_replies() -> None:
    golden = _golden_replies()
    sim = _sim()
    seen: set[str] = set()
    for line in _lines(GOLDEN / "requests.jsonl"):
        req = decode_request(line)
        assert req.id in golden, f"no golden reply for request {req.id!r}"
        _compare(_real_reply(sim, line), golden[req.id], req.id)
        seen.add(json.loads(line)["cmd"])
    assert seen == COMMANDS


def _doc_examples() -> list[tuple[bytes, bytes]]:
    """(request, reply) pairs from the ``>`` / ``<`` lines of the code blocks in protocol.md."""
    pairs: list[tuple[bytes, bytes]] = []
    request: bytes | None = None  # a ">" line still waiting for its "<" reply
    reply: list[str] = []
    for raw in [*DOC.read_text(encoding="utf-8").splitlines(), "```"]:
        if reply and not raw.startswith("    "):
            assert request is not None
            pairs.append((request, "".join(reply).encode()))
            request, reply = None, []
        if raw.startswith("> "):
            assert request is None, f"request without a reply: {request!r}"
            request = raw[2:].encode()
        elif raw.startswith("< "):
            assert request is not None, f"reply without a request: {raw!r}"
            reply = [raw[2:].strip()]
        elif reply:
            reply.append(raw.strip())
    assert request is None, f"request without a reply: {request!r}"
    return pairs


def test_every_documented_example_matches_the_real_reply() -> None:
    pairs = _doc_examples()
    request_lines = [ln for ln in DOC.read_text(encoding="utf-8").splitlines() if ln[:2] == "> "]
    assert len(pairs) == len(request_lines)
    for request, documented in pairs:
        where = request.decode()
        _compare(_real_reply(_sim(), request), documented, where)
    assert {json.loads(req)["cmd"] for req, _ in pairs} == COMMANDS
