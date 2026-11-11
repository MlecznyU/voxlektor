"""Small value types shared by the pure core."""

from typing import Literal, NewType

VoiceName = NewType("VoiceName", str)
EngineId = NewType("EngineId", str)
ChunkId = NewType("ChunkId", int)
Epoch = NewType("Epoch", int)

Lang = Literal["pl", "en"]
LangPref = Literal["auto", "pl", "en"]

Seconds = float
