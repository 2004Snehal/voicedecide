"""Decider: embed a clip once, answer every question from that one embedding."""

from __future__ import annotations

import os
import time
import warnings
from typing import Mapping, Sequence

import numpy as np

from .audio import SR, AudioInput, load_audio
from .errors import ModelRevisionWarning, OutOfSetError
from .model import MODEL_ID, EncoderLike, get_encoder
from .questions import Choice
from .results import DecisionResult


def _source(audio) -> str | None:
    return os.fspath(audio) if isinstance(audio, (str, os.PathLike)) else None


class Decider:
    """Answers questions about audio with EmbeddingGemma 2. No training, no text generation.

    The model downloads from Hugging Face on first use (~2.5 GB, cached afterwards).
    ``strict=True`` raises OutOfSetError instead of returning ``out_of_set=True`` answers.
    """

    def __init__(self, model_id: str = MODEL_ID, *, revision: str | None = None, device: str | None = None,
                 dtype=None, model_path: str | None = None, cache_dir: str | None = None, strict: bool = False,
                 encoder: EncoderLike | None = None):
        self.encoder = encoder or get_encoder(model_id, revision, device, dtype, model_path, cache_dir)
        self.strict = strict
        self._warned_revision: set[int] = set()

    def _check(self, questions: Mapping[str, Choice]):
        if not questions:
            raise ValueError("pass at least one question, e.g. {'route': Choice.from_descriptions({...})}")
        for name, q in questions.items():
            if not isinstance(q, Choice):
                raise TypeError(f"question '{name}' must be a Choice, got {type(q).__name__}")

    def _check_revision(self, questions: Mapping[str, Choice]):
        # runs after embedding, when the loaded snapshot's revision is known
        rev = self.encoder.revision
        for name, q in questions.items():
            if q.mode == "proto" and q.model_revision and rev and q.model_revision != rev and id(q) not in self._warned_revision:
                self._warned_revision.add(id(q))
                warnings.warn(f"question '{name}' was built with model revision {q.model_revision[:10]} but "
                              f"{rev[:10]} is loaded; prototypes may not match. Rebuild the pack or pass "
                              f"Decider(revision='{q.model_revision}').", ModelRevisionWarning, stacklevel=3)

    def _answer(self, emb: np.ndarray, questions: Mapping[str, Choice]) -> list[dict]:
        per_q = {}
        for name, q in questions.items():
            per_q[name] = q.answers_from_scores(q.scores(emb, self.encoder))
            if self.strict:
                for a in per_q[name]:
                    if a.out_of_set:
                        raise OutOfSetError(name, a.similarity, q.min_similarity)
        return [{name: per_q[name][i] for name in questions} for i in range(len(emb))]

    def decide(self, audio: AudioInput, questions: Mapping[str, Choice], *, sampling_rate: int | None = None,
               max_seconds: float | None = None) -> DecisionResult:
        """Decide on one clip. ``audio``: path, bytes, file object, (array, sr), or array + sampling_rate.
        ``max_seconds`` decides on the start of the clip only (lower latency for routing)."""
        self._check(questions)
        t0 = time.perf_counter()
        x = load_audio(audio, sampling_rate, max_seconds)
        t1 = time.perf_counter()
        emb = self.encoder.embed_audio([x], batch_size=1)
        self.encoder.synchronize()
        t2 = time.perf_counter()
        self._check_revision(questions)
        answers = self._answer(emb, questions)[0]
        t3 = time.perf_counter()
        return DecisionResult(answers=answers, duration_s=len(x) / SR, model=self.encoder.model_id,
                              model_revision=self.encoder.revision,
                              source=_source(audio),
                              latency_ms={"decode": (t1 - t0) * 1e3, "embed": (t2 - t1) * 1e3,
                                          "decide": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3})

    def decide_batch(self, audios: Sequence[AudioInput], questions: Mapping[str, Choice], *, batch_size: int = 8,
                     max_seconds: float | None = None) -> list[DecisionResult]:
        """Decide on many clips with batched embedding (offline pipelines). Latency is per batch."""
        self._check(questions)
        results = []
        audios = list(audios)
        for start in range(0, len(audios), batch_size):
            chunk = audios[start:start + batch_size]
            t0 = time.perf_counter()
            xs = [load_audio(a, max_seconds=max_seconds) for a in chunk]
            t1 = time.perf_counter()
            emb = self.encoder.embed_audio(xs, batch_size=batch_size)
            self.encoder.synchronize()
            t2 = time.perf_counter()
            self._check_revision(questions)
            answers = self._answer(emb, questions)
            t3 = time.perf_counter()
            lat = {"decode": (t1 - t0) * 1e3 / len(xs), "embed": (t2 - t1) * 1e3 / len(xs),
                   "decide": (t3 - t2) * 1e3 / len(xs), "total": (t3 - t0) * 1e3 / len(xs)}
            for a, x, ans in zip(chunk, xs, answers):
                results.append(DecisionResult(answers=ans, duration_s=len(x) / SR, model=self.encoder.model_id,
                                              model_revision=self.encoder.revision, latency_ms=dict(lat),
                                              source=_source(a)))
        return results
