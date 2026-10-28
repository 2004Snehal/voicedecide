"""Lazy wrapper around EmbeddingGemma 2 (weights are downloaded from Hugging Face on first use)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import SR

MODEL_ID = "google/embeddinggemma-2"
CLASSIFICATION_PREFIX = "task: classification | query: "
DIM = 768


class EncoderLike(Protocol):
    """What Decider and Choice need from an encoder (lets tests swap in a fake one)."""

    model_id: str
    dim: int

    @property
    def revision(self) -> str | None: ...

    def embed_audio(self, audios: list[np.ndarray], batch_size: int = 8) -> np.ndarray: ...

    def embed_text(self, texts: list[str], prefix: str = CLASSIFICATION_PREFIX) -> np.ndarray: ...

    def synchronize(self) -> None: ...


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Encoder:
    """EmbeddingGemma 2 audio + text towers, loaded on first use.

    dtype: bfloat16 on GPUs with native support (compute capability 8.0+, e.g. A10, L4, A100),
    float32 everywhere else. float16 is refused: it returns NaN or silently degraded embeddings
    for this model.
    """

    def __init__(self, model_id: str = MODEL_ID, revision: str | None = None, device: str | None = None,
                 dtype=None, model_path: str | None = None, cache_dir: str | None = None):
        self.model_id = model_id
        self.dim = DIM
        self._revision = revision
        self._device = device
        self._dtype = dtype
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    # ── loading ──
    def _pick_device_dtype(self):
        import torch

        if self._dtype is not None and self._dtype == torch.float16:
            raise ValueError("float16 is not supported by EmbeddingGemma 2; use bfloat16 or float32")
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._dtype
        if dtype is None:
            bf16_ok = device.startswith("cuda") and torch.cuda.get_device_capability(torch.device(device))[0] >= 8
            dtype = torch.bfloat16 if bf16_ok else torch.float32
        return device, dtype

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    device, dtype = self._pick_device_dtype()
                    self._model = SentenceTransformer(
                        self._model_path or self.model_id, revision=self._revision, device=device,
                        cache_folder=self._cache_dir, model_kwargs={"dtype": dtype},
                        config_kwargs={"vision_config": None},  # audio + text only; saves ~170M params
                    )
        return self._model

    def load(self) -> "Encoder":
        _ = self.model
        return self

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def revision(self) -> str | None:
        """Commit hash of the loaded snapshot (from the HF cache path), or the requested revision."""
        if self._model is None:
            return self._revision
        rev = getattr(getattr(self._model, "model_card_data", None), "base_model_revision", None)
        if rev:
            return rev
        try:  # fall back to the snapshot folder name in the HF cache: .../snapshots/<commit>/config.json
            from huggingface_hub import try_to_load_from_cache

            path = try_to_load_from_cache(self.model_id, "config.json", revision=self._revision)
            if isinstance(path, str) and Path(path).parent.parent.name == "snapshots":
                return Path(path).parent.name
        except Exception:
            pass
        return self._revision

    def synchronize(self) -> None:
        import torch

        if self._model is not None and self.device.startswith("cuda"):
            torch.cuda.synchronize()

    # ── embedding ──
    def embed_audio(self, audios: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """Mono 16 kHz float32 arrays -> (n, 768) L2-normalized float32. Audio takes no prefix."""
        if not audios:
            return np.zeros((0, self.dim), np.float32)
        inputs = [{"array": a, "sampling_rate": SR} for a in audios]
        out = self.model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_tensor=True, show_progress_bar=False)
        emb = out.float().cpu().numpy()
        if not np.isfinite(emb).all():
            raise RuntimeError("model returned non-finite embeddings (is the model running in float16?)")
        return _l2n(emb).astype(np.float32)

    def embed_text(self, texts: list[str], prefix: str = CLASSIFICATION_PREFIX) -> np.ndarray:
        out = self.model.encode(list(texts), prompt=prefix, normalize_embeddings=True,
                                convert_to_tensor=True, show_progress_bar=False)
        return _l2n(out.float().cpu().numpy()).astype(np.float32)


_DEFAULTS: dict[tuple, Encoder] = {}
_DEFAULTS_LOCK = threading.Lock()


def get_encoder(model_id: str = MODEL_ID, revision: str | None = None, device: str | None = None,
                dtype=None, model_path: str | None = None, cache_dir: str | None = None) -> Encoder:
    """Shared encoder per configuration, so Decider and Choice never load the model twice."""
    key = (model_id, revision, device, str(dtype), model_path, cache_dir)
    with _DEFAULTS_LOCK:
        if key not in _DEFAULTS:
            _DEFAULTS[key] = Encoder(model_id, revision, device, dtype, model_path, cache_dir)
        return _DEFAULTS[key]
