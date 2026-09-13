"""Microphone capture, resampling, and in-memory WAV encoding."""

from __future__ import annotations

import io
import queue
import threading
import wave
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000


def to_mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = x.T
        return np.mean(x, axis=-1).astype(np.float32)
    return x.reshape(-1).astype(np.float32)


def resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    x = to_mono(audio)
    if sr == SAMPLE_RATE or x.size == 0:
        return x.astype(np.float32, copy=False)
    n = max(1, int(round(x.size * SAMPLE_RATE / float(sr))))
    old_idx = np.linspace(0.0, 1.0, x.size, endpoint=False)
    new_idx = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def float_pcm(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    return np.clip(x, -1.0, 1.0)


def pcm_to_wav_bytes(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    x = float_pcm(pcm)
    i16 = np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(i16.tobytes())
    return buf.getvalue()


def load_wav_file(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return resample_16k(x, sr)


class MicStream:
    """WASAPI/shared-mode capture into a lock-free PCM queue, always 16 kHz mono."""

    def __init__(self, device: Optional[int] = None, block_ms: int = 20):
        self.device = device
        self.block_ms = max(10, int(block_ms))
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stream = None
        self._in_sr = SAMPLE_RATE
        self._lock = threading.Lock()
        self._dropped = 0

    def start(self) -> None:
        import sounddevice as sd

        device = self.device
        info = sd.query_devices(device, "input")
        native_sr = int(info.get("default_samplerate") or SAMPLE_RATE)
        # Prefer native 16 kHz; otherwise capture at device rate and resample.
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=SAMPLE_RATE)
            self._in_sr = SAMPLE_RATE
        except Exception:
            self._in_sr = native_sr

        blocksize = max(64, int(self._in_sr * self.block_ms / 1000.0))

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            x = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32).copy()
            if self._in_sr != SAMPLE_RATE:
                x = resample_16k(x, self._in_sr)
            try:
                self._q.put_nowait(x)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(x)
                except queue.Full:
                    self._dropped += 1

        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self._in_sr,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            callback=callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.2) -> np.ndarray:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return np.zeros((0,), dtype=np.float32)

    def drain(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def list_input_devices() -> list[tuple[int, str, int]]:
    import sounddevice as sd

    out: list[tuple[int, str, int]] = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels") or 0) <= 0:
            continue
        name = str(dev.get("name") or f"device-{i}")
        sr = int(dev.get("default_samplerate") or 0)
        out.append((i, name, sr))
    return out
