from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class NoteEvent:
    """Canonical note contract used across pipelines + UI export.

    Timing is in seconds.
    - pitch: MIDI integer (0-127 typically)
    - onset: seconds from start
    - duration: seconds
    """

    pitch: int
    onset: float
    duration: float
    velocity: int = 80

    # Guitar mapping (assigned later)
    string: Optional[int] = None   # 6 (low E) .. 1 (high E)
    fret: Optional[int] = None

    # Chord grouping (poly)
    chord_id: Optional[int] = None

    @property
    def offset(self) -> float:
        return float(self.onset + self.duration)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
