from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from app.models.note_event import NoteEvent


@dataclass
class BasicPitchConfig:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length: float = 0.05  # seconds
    prefer_cpu: bool = False        # if True, force TensorFlow to use CPU only


def _hz_to_midi(hz: float) -> int:
    # MIDI = 69 + 12*log2(hz/440)
    hz = float(hz)
    if hz <= 0:
        return 0
    return int(round(69.0 + 12.0 * np.log2(hz / 440.0)))


def _row_to_note(row: Any) -> Optional[NoteEvent]:
    """
    Normalize Basic Pitch note_events row into NoteEvent(pitch=MIDI).
    Handles dict or array-like forms.
    """
    try:
        if isinstance(row, dict):
            start = float(row.get("start_time", row.get("start", 0.0)))
            end = float(row.get("end_time", row.get("end", start)))

            # Prefer explicit MIDI fields
            if "pitch_midi" in row:
                pitch_midi = int(round(float(row["pitch_midi"])))
            elif "pitch" in row:
                p = float(row["pitch"])
                # Heuristic: Hz typically > 20, MIDI <= 127
                pitch_midi = _hz_to_midi(p) if p > 127 else int(round(p))
            else:
                return None

            amp = float(row.get("amplitude", 0.7))

        else:
            # array-like: [start, end, pitch, amplitude?]
            start = float(row[0])
            end = float(row[1])

            p = float(row[2])
            pitch_midi = _hz_to_midi(p) if p > 127 else int(round(p))

            amp = float(row[3]) if len(row) > 3 else 0.7

        dur = max(0.0, end - start)
        if dur <= 0:
            return None

        pitch_midi = int(max(0, min(127, pitch_midi)))
        vel = int(max(1, min(127, round(amp * 127))))

        return NoteEvent(pitch=pitch_midi, onset=start, duration=dur, velocity=vel)

    except Exception:
        return None


def transcribe_with_basic_pitch(audio_path: str, cfg: Optional[BasicPitchConfig] = None) -> List[NoteEvent]:
    """
    Polyphonic transcription using Spotify Basic Pitch.
    Returns NoteEvent list sorted by onset then pitch.
    """
    cfg = cfg or BasicPitchConfig()

    try:
        from basic_pitch.inference import predict
    except Exception as e:
        raise RuntimeError(
            "Basic Pitch failed to import. Install with: pip install basic-pitch[tf]\n"
            f"Import error: {e!r}"
        ) from e

    _, _, note_events = predict(
        audio_path,
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        minimum_note_length=cfg.min_note_length,
    )

    if note_events is None:
        return []

    rows = note_events
    try:
        arr = np.asarray(note_events)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            rows = arr
    except Exception:
        pass

    notes: List[NoteEvent] = []
    for r in rows:
        n = _row_to_note(r)
        if n is None:
            continue
        if n.duration < cfg.min_note_length:
            continue
        notes.append(n)

    # Guitar playable-ish sanity filter (optional but helps)
    # E2 (40) .. E6 (88) typical guitar range, keep a bit wider
    notes = [n for n in notes if 30 <= n.pitch <= 96]

    notes.sort(key=lambda x: (x.onset, x.pitch))
    return notes


# Compatibility export if any module expects this name
transcribe_basic_pitch_poly = transcribe_with_basic_pitch

__all__ = ["BasicPitchConfig", "transcribe_with_basic_pitch", "transcribe_basic_pitch_poly"]