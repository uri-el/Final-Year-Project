from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple, Dict

from app.models.note_event import NoteEvent


# Standard guitar tuning (MIDI)
# String numbers: 6 (low E) .. 1 (high E)
STANDARD_TUNING: Dict[int, int] = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}


def _candidate_positions(
    pitch: int,
    max_fret: int = 20,
    tuning: Dict[int, int] = STANDARD_TUNING,
) -> List[Tuple[int, int]]:
    """
    Return all (string, fret) positions that can play `pitch`.
    """
    cands: List[Tuple[int, int]] = []
    for s in (6, 5, 4, 3, 2, 1):
        open_midi = tuning[s]
        fret = pitch - open_midi
        if 0 <= fret <= max_fret:
            cands.append((s, fret))
    return cands


def _choose_position(
    pitch: int,
    prev_string: Optional[int],
    prev_fret: Optional[int],
    max_fret: int = 20,
) -> Optional[Tuple[int, int]]:
    """
    Heuristic choice:
      - prefer mid-neck position (around fret 5) to avoid too many open strings
      - prefer continuity with previous fret to reduce jumping
    """
    cands = _candidate_positions(pitch, max_fret=max_fret)
    if not cands:
        return None

    def score(sf: Tuple[int, int]) -> Tuple[int, int, int]:
        s, f = sf

        # 1) Bias toward mid-neck (around fret 5)
        ideal_position = abs(f - 5)

        # 2) Continuity with previous fret (smooth movement)
        continuity = abs(f - (prev_fret if prev_fret is not None else f))

        # 3) Small preference for staying on nearby string (minor tie-breaker)
        string_jump = abs(s - (prev_string if prev_string is not None else s))

        return (ideal_position, continuity, string_jump)

    cands.sort(key=score)
    return cands[0]


def map_notes_to_guitar(
    notes: List[NoteEvent],
    max_fret: int = 20,
) -> List[NoteEvent]:
    """
    Assign string/fret for each note deterministically.
    Leaves notes unchanged if mapping isn't possible.
    """
    out: List[NoteEvent] = []

    prev_string: Optional[int] = None
    prev_fret: Optional[int] = None

    # Stable mapping order
    notes_sorted = sorted(notes, key=lambda n: (n.onset, n.pitch))

    for n in notes_sorted:
        pos = _choose_position(n.pitch, prev_string, prev_fret, max_fret=max_fret)
        if pos is None:
            out.append(n)
            continue

        s, f = pos
        out.append(replace(n, string=s, fret=f))

        prev_string, prev_fret = s, f

    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


# --- Backward-compatible aliases for older imports ---
def map_to_guitar(notes: List[NoteEvent], max_fret: int = 20) -> List[NoteEvent]:
    """
    Alias used by older chord solver modules.
    """
    return map_notes_to_guitar(notes, max_fret=max_fret)


__all__ = ["STANDARD_TUNING", "map_notes_to_guitar", "map_to_guitar"]