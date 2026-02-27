# app/live/audio_stream.py
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _sd_import_error = e


@dataclass
class AudioStreamConfig:
    sample_rate: int = 44100
    channels: int = 1
    dtype: str = "float32"
    buffer_seconds: float = 12.0


class RingBufferAudio:
    """
    Thread-safe ring buffer storing recent audio chunks.

    stream_time() is sample-clock based (captured samples / sample_rate),
    which keeps onset alignment stable.
    """
    def __init__(self, sample_rate: int, max_seconds: float):
        self.sample_rate = int(sample_rate)
        self.max_samples = int(self.sample_rate * max_seconds)

        self._buf: Deque[np.ndarray] = deque()
        self._buf_samples = 0
        self._total_samples = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)
        chunk = chunk.astype(np.float32, copy=False)

        with self._lock:
            self._buf.append(chunk)
            self._buf_samples += chunk.shape[0]
            self._total_samples += chunk.shape[0]

            while self._buf_samples > self.max_samples and self._buf:
                old = self._buf.popleft()
                self._buf_samples -= old.shape[0]

    def get_last_seconds(self, seconds: float) -> np.ndarray:
        need = int(self.sample_rate * float(seconds))
        with self._lock:
            if self._buf_samples == 0:
                return np.zeros((0,), dtype=np.float32)

            take = min(need, self._buf_samples)
            remaining = take
            chunks = []

            for arr in reversed(self._buf):
                if remaining <= 0:
                    break
                if arr.shape[0] <= remaining:
                    chunks.append(arr)
                    remaining -= arr.shape[0]
                else:
                    chunks.append(arr[-remaining:])
                    remaining = 0

            if not chunks:
                return np.zeros((0,), dtype=np.float32)

            return np.concatenate(list(reversed(chunks))).astype(np.float32, copy=False)

    def stream_time(self) -> float:
        with self._lock:
            return float(self._total_samples) / float(self.sample_rate)


class LiveAudioInput:
    """
    Continuous mic capture -> ring buffer.
    Uses device default samplerate (common fix on Windows).
    """
    def __init__(self, cfg: AudioStreamConfig):
        self.cfg = cfg
        self.ring = RingBufferAudio(cfg.sample_rate, cfg.buffer_seconds)
        self._stream = None
        self._running = False

    def start(self, device: Optional[int] = None) -> None:
        if sd is None:
            raise RuntimeError(f"sounddevice import failed: {_sd_import_error!r}")
        if self._running:
            return

        if device is not None:
            info = sd.query_devices(device, "input")
            dev_sr = int(round(float(info.get("default_samplerate", self.cfg.sample_rate))))
            self.cfg.sample_rate = dev_sr
            self.ring = RingBufferAudio(self.cfg.sample_rate, self.cfg.buffer_seconds)

        def callback(indata, frames, time_info, status):
            x = indata
            if x.ndim == 2 and x.shape[1] > 1:
                x = np.mean(x, axis=1)
            else:
                x = x.reshape(-1)
            self.ring.push(x)

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            callback=callback,
            device=device,
            blocksize=0,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self._running = False


__all__ = ["AudioStreamConfig", "LiveAudioInput", "RingBufferAudio"]