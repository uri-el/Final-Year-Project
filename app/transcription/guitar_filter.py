# app/transcription/guitar_filter.py
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from app.models.note_event import NoteEvent

GUITAR_MIDI_MIN = 40  # E2
GUITAR_MIDI_MAX = 88  # E6


def guitar_only_pipeline(notes: Iterable[NoteEvent]) -> List[NoteEvent]:
    """
    Keep only plausible guitar notes (range + simple duration + dedupe).
    """
    xs: List[NoteEvent] = []
    for n in notes:
        p = int(n.pitch)
        if GUITAR_MIDI_MIN <= p <= GUITAR_MIDI_MAX and float(n.duration) >= 0.06:
            xs.append(n)

    xs.sort(key=lambda n: (float(n.onset), int(n.pitch)))

    out: List[NoteEvent] = []
    last: Optional[Tuple[float, int]] = None
    for n in xs:
        key = (float(n.onset), int(n.pitch))
        if last is not None:
            if abs(key[0] - last[0]) <= 0.05 and abs(key[1] - last[1]) <= 0:
                continue
        out.append(n)
        last = key

    return out