from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------- Pitch tracking (continuous MIDI float) ----------

def _midi_f(hz: float) -> float:
    return float(69.0 + 12.0 * np.log2(hz / 440.0))


def yin_pitch_hz(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 900.0) -> Optional[float]:
    """
    Continuous YIN pitch (Hz). Returns None if no reliable pitch.
    This is a trimmed, offline-friendly version.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < 512:
        return None

    x = x - np.mean(x)

    max_tau = int(sr / fmin)
    min_tau = int(sr / fmax)
    max_tau = min(max_tau, x.size - 1)
    if max_tau <= min_tau + 2:
        return None

    d = np.zeros(max_tau + 1, dtype=np.float32)
    for tau in range(1, max_tau + 1):
        diff = x[:-tau] - x[tau:]
        d[tau] = np.sum(diff * diff)

    cmnd = np.zeros_like(d)
    cmnd[0] = 1.0
    running = 0.0
    for tau in range(1, max_tau + 1):
        running += float(d[tau])
        cmnd[tau] = (d[tau] * tau) / (running + 1e-9)

    thresh = 0.20
    tau = None
    for t in range(min_tau, max_tau):
        if cmnd[t] < thresh:
            tau = t
            break
    if tau is None:
        tau = int(np.argmin(cmnd[min_tau:max_tau]) + min_tau)

    while tau + 1 < max_tau and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # parabolic interpolation
    t0 = max(min_tau, tau - 1)
    t1 = tau
    t2 = min(max_tau - 1, tau + 1)
    y0, y1, y2 = float(cmnd[t0]), float(cmnd[t1]), float(cmnd[t2])
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        tau_hat = float(t1)
    else:
        tau_hat = float(t1) + 0.5 * (y0 - y2) / denom

    if tau_hat <= 0:
        return None

    hz = float(sr / tau_hat)

    # fold down harmonics a bit (helps guitar)
    while hz > 520.0:
        hz *= 0.5

    if hz < 55.0 or hz > 1200.0:
        return None

    return hz


def track_contour(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 10.0,
    rms_gate: float = 0.0002,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times_s, midi_float (nan when unvoiced), rms, onset_strength
    onset_strength is simple RMS derivative (good enough for hammer vs pick).
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(int(sr * frame_ms / 1000.0), 1024)
    hop = max(int(sr * hop_ms / 1000.0), 256)

    times = []
    midi_f = []
    rms = []

    for start in range(0, max(0, audio.size - frame), hop):
        x = audio[start:start + frame]
        t = start / sr
        r = float(np.sqrt(np.mean(x * x))) if x.size else 0.0

        hz = None
        if r >= rms_gate:
            hz = yin_pitch_hz(x, sr)

        times.append(t)
        rms.append(r)
        midi_f.append(np.nan if hz is None else _midi_f(hz))

    times = np.asarray(times, dtype=np.float32)
    rms = np.asarray(rms, dtype=np.float32)
    midi_f = np.asarray(midi_f, dtype=np.float32)

    # onset strength: positive RMS derivative (rectified)
    dr = np.diff(rms, prepend=rms[:1])
    onset = np.maximum(0.0, dr).astype(np.float32)

    return times, midi_f, rms, onset


# ---------- Annotation over Basic Pitch notes ----------

def _slice_by_time(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.where((times >= t0) & (times <= t1))[0]


def _robust_median(x: np.ndarray) -> Optional[float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.median(x))


def _is_monotonic(y: np.ndarray, min_frac: float = 0.7) -> bool:
    y = y[np.isfinite(y)]
    if y.size < 6:
        return False
    dy = np.diff(y)
    pos = np.sum(dy > 0)
    neg = np.sum(dy < 0)
    return (pos / max(1, dy.size) >= min_frac) or (neg / max(1, dy.size) >= min_frac)


def annotate_articulations(
    *,
    audio: np.ndarray,
    sr: int,
    notes: List[Dict],   # your pipeline dicts: pitch,onset_s,dur_s,string,fret
) -> List[Dict]:
    """
    Adds:
      note["tech"] = {"type": "bend"/"slide"/"hammer"/"pull", ...}
    Only annotates when confident.
    """
    times, midi_f, rms, onset = track_contour(audio, sr)

    # Helper for onset strength around a time (peak in +-30ms)
    def onset_peak(t: float, w: float = 0.03) -> float:
        idx = _slice_by_time(times, t - w, t + w)
        if idx.size == 0:
            return 0.0
        return float(np.max(onset[idx]))

    out = [dict(n) for n in notes]
    out.sort(key=lambda r: float(r["onset_s"]))

    # First annotate bends/slides per note
    for n in out:
        t0 = float(n["onset_s"])
        t1 = t0 + float(n["dur_s"])
        idx = _slice_by_time(times, t0, t1)
        if idx.size < 8:
            continue

        y = midi_f[idx]
        y = y[np.isfinite(y)]
        if y.size < 8:
            continue

        # stable start/end medians
        start_idx = _slice_by_time(times, t0, min(t1, t0 + 0.08))
        end_idx = _slice_by_time(times, max(t0, t1 - 0.08), t1)

        y0 = _robust_median(midi_f[start_idx]) if start_idx.size else None
        y1 = _robust_median(midi_f[end_idx]) if end_idx.size else None
        if y0 is None or y1 is None:
            continue

        span = float(np.nanmax(y) - np.nanmin(y))

        # BEND: within-note drift >= ~0.75 semitone and mostly monotonic
        if span >= 0.75 and _is_monotonic(y, min_frac=0.65):
            n["tech"] = {
                "type": "bend",
                "from": float(y0),
                "to": float(y1),
                "semitones": float(y1 - y0),
            }
            continue

        # SLIDE: start and end differ by >=2 semitones and contour is smooth-ish
        if abs(y1 - y0) >= 2.0 and _is_monotonic(y, min_frac=0.65):
            n["tech"] = {
                "type": "slide",
                "from": float(y0),
                "to": float(y1),
                "semitones": float(y1 - y0),
            }
            continue

    # Then annotate hammer/pull based on weak onsets between consecutive notes
    for i in range(len(out) - 1):
        a = out[i]
        b = out[i + 1]

        gap = float(b["onset_s"]) - (float(a["onset_s"]) + float(a["dur_s"]))
        # legato window: next note starts very soon after previous ends (or overlaps a bit)
        if gap > 0.20:
            continue

        oa = onset_peak(float(a["onset_s"]))
        ob = onset_peak(float(b["onset_s"]))

        # If next onset is much weaker, treat as legato
        # (ratio-based so it scales across recordings)
        if oa <= 1e-6:
            continue
        if ob / oa > 0.45:
            continue

        dp = int(b["pitch"]) - int(a["pitch"])
        if dp >= 1:
            b["tech"] = {"type": "hammer", "from": int(a["fret"]), "to": int(b["fret"])}
        elif dp <= -1:
            b["tech"] = {"type": "pull", "from": int(a["fret"]), "to": int(b["fret"])}

    return out