"""Fake encoder: tones of different pitch play the role of different classes.

Audio of a tone near FREQS[k] embeds close to class vector V[k]; text containing "tone k"
embeds close to V[k] as well, so text-anchor and prototype modes can both be tested without
downloading the real model.
"""

import hashlib

import numpy as np
import pytest

from voicedecide import model as vd_model
from voicedecide.audio import SR

DIM = 768
FREQS = [220.0, 330.0, 440.0, 550.0, 660.0]
_rng = np.random.default_rng(0)
V = _rng.normal(size=(len(FREQS), DIM))
V /= np.linalg.norm(V, axis=1, keepdims=True)
NOISE_DIR = _rng.normal(size=DIM)
NOISE_DIR /= np.linalg.norm(NOISE_DIR)


def _noise(key: bytes, scale: float) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=DIM) * scale / np.sqrt(DIM)


class FakeEncoder:
    model_id = "fake/encoder"
    dim = DIM
    revision = "fakerev0001"

    def __init__(self):
        self.audio_calls = 0

    def embed_audio(self, audios, batch_size=8):
        self.audio_calls += len(audios)
        out = []
        for x in audios:
            spec = np.abs(np.fft.rfft(x))
            f = np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]
            k = int(np.argmin([abs(f - g) for g in FREQS]))
            if abs(f - FREQS[k]) > 40:  # not one of the known tones: lands far from every class
                v = NOISE_DIR + _noise(x.tobytes(), 0.5)
            else:
                v = V[k] + _noise(x.tobytes(), 0.6)
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)

    def embed_text(self, texts, prefix=""):
        out = []
        for t in texts:
            k = next((i for i in range(len(FREQS)) if f"tone {i}" in t), None)
            v = (V[k] if k is not None else NOISE_DIR) + _noise(t.encode(), 0.9)
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)

    def synchronize(self):
        pass


def tone(k: int, seconds: float = 1.0, sr: int = SR, seed: int = 0, freq: float | None = None) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(seed)
    f = FREQS[k] if freq is None else freq
    return (0.5 * np.sin(2 * np.pi * (f + rng.uniform(-5, 5)) * t) + 0.01 * rng.normal(size=t.size)).astype(np.float32)


@pytest.fixture
def fake():
    enc = FakeEncoder()
    return enc


@pytest.fixture(autouse=True)
def _patch_default_encoder(monkeypatch, fake):
    """Every Decider()/Choice built without an explicit encoder uses the fake one."""
    monkeypatch.setattr(vd_model, "get_encoder", lambda *a, **k: fake)
    import voicedecide.decide as dec
    import voicedecide.questions as q

    monkeypatch.setattr(dec, "get_encoder", lambda *a, **k: fake)
    monkeypatch.setattr(q, "get_encoder", lambda *a, **k: fake)
    yield


@pytest.fixture
def examples():
    """{label: [(array, sr), ...]} with 6 clips per class for 3 classes."""
    return {f"c{k}": [(tone(k, seed=100 * k + i), SR) for i in range(6)] for k in range(3)}


@pytest.fixture
def val():
    audio, labels = [], []
    for k in range(3):
        for i in range(12):
            audio.append((tone(k, seed=1000 + 100 * k + i), SR))
            labels.append(f"c{k}")
    return audio, labels
