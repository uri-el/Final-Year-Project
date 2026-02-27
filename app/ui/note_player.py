from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import soundfile as sf


@dataclass
class SynthNote:
    pitch: int
    onset_s: float
    dur_s: float


def midi_to_freq(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _karplus_strong(freq_hz: float, dur_s: float, sr: int) -> np.ndarray:
    """
    Simple plucked-string synthesis.
    Produces a much more guitar-like timbre than additive sines.
    """
    n = int(max(1, dur_s * sr))
    # delay length
    N = int(sr / max(40.0, float(freq_hz)))
    N = max(2, N)

    # brightness: higher notes a bit brighter, lower notes slightly darker
    # control decay as a function of pitch
    # (lower decay -> faster damping)
    if freq_hz < 110:
        decay = 0.992
    elif freq_hz < 220:
        decay = 0.994
    else:
        decay = 0.996

    # initial noise burst = "pluck"
    buf = (np.random.rand(N).astype(np.float32) * 2.0 - 1.0) * 0.6

    y = np.zeros(n, dtype=np.float32)
    idx = 0
    for i in range(n):
        y[i] = buf[idx]
        nxt = (idx + 1) % N
        # averaging filter with decay = string loss
        buf[idx] = decay * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt

    # pick attack envelope (fast attack, then natural decay)
    a = int(0.004 * sr)  # 4ms
    if a > 1 and y.size >= a:
        y[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)

    # optional extra damping tail so notes stop cleanly
    r = int(0.020 * sr)  # 20ms release
    if r > 1 and y.size >= r:
        y[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)

    return y


def synth_notes_to_wav_path(
    notes: Iterable[SynthNote],
    *,
    sr: int = 44100,
    gain: float = 0.35,
) -> Optional[str]:
    notes = list(notes)
    if not notes:
        return None

    end_t = max(float(n.onset_s) + float(n.dur_s) for n in notes)
    if end_t <= 0:
        return None

    n_samples = int(end_t * sr) + 1
    y = np.zeros((n_samples,), dtype=np.float32)

    for n in notes:
        dur = float(n.dur_s)
        if dur <= 0:
            continue

        # clamp tiny notes
        dur = max(dur, 0.06)

        start = int(float(n.onset_s) * sr)
        if start >= y.shape[0]:
            continue

        f = midi_to_freq(int(n.pitch))

        # Slightly extend synthesis duration so tail isn't cut abruptly
        wave = _karplus_strong(f, dur_s=dur * 1.05, sr=sr)

        end = min(start + wave.shape[0], y.shape[0])
        y[start:end] += wave[: (end - start)]

    # normalize
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 1e-9:
        y = (float(gain) / peak) * y

    # avoid clipping after normalization (extra safety)
    y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr)
    return path