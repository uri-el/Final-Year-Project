from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from app.models.note_event import NoteEvent


def cleanup_notes(
    notes: List[NoteEvent],
    *,
    min_dur: float = 0.12,
    dedupe_onset_tol: float = 0.04,
    pitch_range: Tuple[int, int] = (35, 96),
) -> List[NoteEvent]:
    # range + duration filter
    out = [
        n for n in notes
        if n.duration >= min_dur and pitch_range[0] <= n.pitch <= pitch_range[1]
    ]
    out.sort(key=lambda n: (n.onset, n.pitch))

    # de-dup close repeats of same pitch
    deduped: List[NoteEvent] = []
    last_by_pitch: Dict[int, NoteEvent] = {}
    for n in out:
        prev = last_by_pitch.get(n.pitch)
        if prev is not None and abs(n.onset - prev.onset) <= dedupe_onset_tol:
            # keep louder, earlier onset
            keep = n if n.velocity >= prev.velocity else prev
            keep = replace(keep, onset=min(n.onset, prev.onset))
            last_by_pitch[n.pitch] = keep
            continue
        last_by_pitch[n.pitch] = n
        deduped.append(n)

    deduped.sort(key=lambda n: (n.onset, n.pitch))
    return deduped


def quantize_notes(
    notes: List[NoteEvent],
    *,
    bpm: float,
    grid: str = "1/16",
) -> List[NoteEvent]:
    """
    grid: "1/8" or "1/16"
    """
    if bpm <= 0:
        return notes

    beat = 60.0 / float(bpm)
    if grid == "1/8":
        step = beat / 2.0
    else:
        step = beat / 4.0  # 1/16

    out: List[NoteEvent] = []
    for n in notes:
        q_on = round(n.onset / step) * step
        q_off = round((n.onset + n.duration) / step) * step
        dur = max(step, q_off - q_on)
        out.append(replace(n, onset=float(q_on), duration=float(dur)))

    out.sort(key=lambda n: (n.onset, n.pitch))
    return out