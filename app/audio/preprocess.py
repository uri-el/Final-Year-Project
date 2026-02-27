from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import soundfile as sf


def ensure_wav_mono(
    input_path: str,
    target_sr: int = 22050,
) -> str:
    """
    Returns a path to a WAV (mono, target_sr). Creates a temp wav if needed.
    Requires ffmpeg for mp3 (and other non-wav formats).
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".wav":
        # still normalize to mono + target_sr if needed
        audio, sr = sf.read(input_path, always_2d=False)
        audio = to_mono(audio)
        if sr != target_sr:
            audio = resample_linear(audio, sr, target_sr)
            sr = target_sr
        out = _temp_wav_path()
        sf.write(out, audio.astype(np.float32, copy=False), sr)
        return out

    # Anything else (mp3 etc) -> ffmpeg convert
    out = _temp_wav_path()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(target_sr),
        out
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH.\n"
            "Then restart your terminal/IDE."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg failed converting input to wav.") from e

    return out


def to_mono(x):
    import numpy as np
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.mean(x, axis=1)
    return x


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    n_in = int(x.shape[0])
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_out <= 8:
        return x
    t_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32, copy=False)


def _temp_wav_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path