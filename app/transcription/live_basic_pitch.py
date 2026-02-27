from __future__ import annotations

import os
import tempfile
import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.models.note_event import NoteEvent
from app.transcription.basic_pitch_poly import BasicPitchConfig, transcribe_with_basic_pitch


def _resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n_in = int(audio.shape[0])
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_out <= 8:
        return audio.astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def transcribe_window(audio: np.ndarray, capture_sr: int, cfg: BasicPitchConfig) -> List[NoteEvent]:
    """
    Write window to a temp wav and run Basic Pitch.
    Resamples to 22050 for Basic Pitch stability.
    """
    audio = np.nan_to_num(audio).astype(np.float32, copy=False)
    audio = np.clip(audio, -1.0, 1.0)

    target_sr = 22050
    if capture_sr != target_sr:
        audio = _resample_linear(audio, capture_sr, target_sr)
        sr = target_sr
    else:
        sr = capture_sr

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, sr)
        return transcribe_with_basic_pitch(path, cfg=cfg)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def _note_key(n: NoteEvent, onset_tol: float = 0.06) -> Tuple[int, float]:
    """
    Quantized key for deduplication: (pitch, rounded_onset).
    Two notes with same pitch and onset within onset_tol are considered the same.
    """
    bucket = round(n.onset / onset_tol) * onset_tol
    return (int(n.pitch), round(bucket, 3))


def live_loop_basic_pitch(
    *,
    get_audio_window: Callable[[float], np.ndarray],
    get_stream_time: Callable[[], float],
    capture_sr: int,
    on_new: Callable[[List[NoteEvent]], None],
    should_stop: Callable[[], bool],
    window_sec: float = 3.0,
    hop_sec: float = 0.5,
    cfg: Optional[BasicPitchConfig] = None,
    rms_gate: float = 0.0003,
) -> None:
    """
    Near-real-time polyphonic loop using Basic Pitch.

    Fixes over previous version:
      1) Larger window (3s) with more overlap so edge notes get a second
         chance in the next window.
      2) Pitch-aware deduplication instead of onset watermark — chords and
         fast runs no longer get swallowed.
      3) Lower RMS gate (0.0003) so soft fingerpicking is not skipped.
      4) Hop adapts to inference time: we measure how long Basic Pitch takes
         and sleep only the remainder, avoiding drift.
    """
    cfg = cfg or BasicPitchConfig()

    # Track already-emitted notes by (pitch, quantized_onset) to prevent repeats
    emitted: Dict[Tuple[int, float], float] = {}  # key -> wall_time when emitted
    DEDUP_TOL = 0.06   # seconds: notes within this onset window are "same note"
    HISTORY_SEC = 10.0  # prune emit history older than this

    while not should_stop():
        t_now = float(get_stream_time())

        if t_now < window_sec:
            time.sleep(hop_sec)
            continue

        audio = get_audio_window(window_sec)
        if audio.size < int(capture_sr * window_sec * 0.3):
            time.sleep(hop_sec)
            continue

        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        if rms < rms_gate:
            time.sleep(hop_sec)
            continue

        t_before = time.monotonic()
        notes = transcribe_window(audio, capture_sr, cfg)
        inference_time = time.monotonic() - t_before

        # Shift local window onsets to global stream time
        window_start = t_now - window_sec
        shifted = [replace(n, onset=float(n.onset + window_start)) for n in notes]
        shifted.sort(key=lambda n: (n.onset, n.pitch))

        # Only consider notes whose onset falls in the recent hop region
        # (with margin to catch edge notes the previous window may have clipped)
        margin = 0.3
        emit_start = t_now - hop_sec - margin
        candidates = [
            n for n in shifted
            if emit_start <= float(n.onset) <= t_now
        ]

        # Deduplicate against already-emitted notes (pitch + quantized onset)
        new_notes: List[NoteEvent] = []
        for n in candidates:
            key = _note_key(n, DEDUP_TOL)
            if key not in emitted:
                new_notes.append(n)
                emitted[key] = t_now

        if new_notes:
            on_new(new_notes)

        # Prune stale entries
        cutoff = t_now - HISTORY_SEC
        stale = [k for k, t in emitted.items() if t < cutoff]
        for k in stale:
            del emitted[k]

        # Sleep only the remainder after inference so we stay on schedule
        sleep_time = max(0.05, hop_sec - inference_time)
        time.sleep(sleep_time)
