"""Turn any audio input into mono float32 at 16 kHz, the format EmbeddingGemma 2 expects."""

from __future__ import annotations

import hashlib
import io
import os
import warnings
from math import gcd
from typing import Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import AudioTruncatedWarning

SR = 16_000
# 25 audio tokens per second in an 8,192-token context gives ~327 s; keep a little headroom.
MAX_SECONDS = 320

AudioInput = Union[str, "os.PathLike[str]", bytes, bytearray, io.IOBase, np.ndarray, tuple, dict]


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    # soundfile returns (frames, channels); arrays from other tools are often (channels, frames)
    channel_axis = 0 if x.shape[0] < x.shape[1] and x.shape[0] <= 8 else 1
    return x.mean(axis=channel_axis)


def _int_to_float(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float32, copy=False)


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(int(sr), target)
    return resample_poly(x, target // g, int(sr) // g).astype(np.float32)


def _read(audio: AudioInput, sampling_rate: int | None) -> tuple[np.ndarray, int]:
    if isinstance(audio, (str, os.PathLike)):
        x, sr = sf.read(os.fspath(audio), dtype="float32", always_2d=True)
        return x, sr
    if isinstance(audio, (bytes, bytearray)):
        x, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        return x, sr
    if isinstance(audio, io.IOBase):
        x, sr = sf.read(audio, dtype="float32", always_2d=True)
        return x, sr
    if isinstance(audio, dict):  # Hugging Face datasets style: {"array": ..., "sampling_rate": ...} or {"bytes": ...}
        if audio.get("bytes") is not None:
            return _read(audio["bytes"], sampling_rate)
        if "array" in audio:
            return _read((audio["array"], audio.get("sampling_rate", sampling_rate)), None)
        if audio.get("path"):
            return _read(audio["path"], sampling_rate)
        raise ValueError("audio dict needs 'array' + 'sampling_rate', 'bytes', or 'path'")
    if isinstance(audio, tuple) and len(audio) == 2:
        return _read(audio[0], audio[1])
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().cpu().numpy()
    if isinstance(audio, np.ndarray):
        if sampling_rate is None:
            raise ValueError("raw arrays need a sampling rate: pass (array, sr) or sampling_rate=...")
        return _int_to_float(np.asarray(audio)), int(sampling_rate)
    raise TypeError(f"unsupported audio input type: {type(audio).__name__}")


def load_audio(audio: AudioInput, sampling_rate: int | None = None, max_seconds: float | None = None) -> np.ndarray:
    """Return mono float32 audio at 16 kHz.

    Accepts a file path, encoded bytes or file object (wav/flac/ogg/mp3), a ``(array, sr)``
    tuple, a NumPy array or torch tensor with ``sampling_rate``, or a Hugging Face audio dict.
    ``max_seconds`` keeps only the start of the clip (e.g. decide on the first 5 s of a call).
    """
    x, sr = _read(audio, sampling_rate)
    x = resample(_to_mono(x), sr)
    if x.size == 0:
        raise ValueError("audio is empty")
    limit = MAX_SECONDS if max_seconds is None else min(max_seconds, MAX_SECONDS)
    n = int(limit * SR)
    if x.size > n:
        if max_seconds is None or max_seconds > MAX_SECONDS:
            warnings.warn(f"audio is {x.size / SR:.0f} s; keeping the first {MAX_SECONDS} s (model context limit)",
                          AudioTruncatedWarning, stacklevel=3)
        x = x[:n]
    return np.ascontiguousarray(x, dtype=np.float32)


def fingerprint(x: np.ndarray) -> str:
    """Stable hash of decoded audio, used to detect validation clips that were also prototypes."""
    return hashlib.sha1(np.round(x, 4).tobytes()).hexdigest()
