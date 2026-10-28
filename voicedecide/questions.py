"""Choice: one question with a fixed set of options, answered from an audio embedding.

Two modes:
* ``text``  (default, zero-shot): options are described in words; audio is compared with the
  averaged text embeddings of those descriptions.
* ``proto`` (recommended, few-shot): options are defined by ~10 labeled clips each; audio is
  compared with the mean embedding (prototype) of each option's clips.
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import calibrate as cal
from .audio import AudioInput, fingerprint, load_audio
from .errors import LeakageWarning, UnknownLabelError, UntestedLanguageWarning, VoiceDecideError, ZeroShotHint
from .languages import untested_message
from .model import CLASSIFICATION_PREFIX, MODEL_ID, EncoderLike, get_encoder
from .results import Answer

_HINT_SHOWN = False
OOS_MARGIN = 0.01   # cosine slack below the weakest validation clip before audio counts as out of set


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _encoder_of(encoder) -> EncoderLike:
    if encoder is None:
        return get_encoder()
    return getattr(encoder, "encoder", encoder)  # accept a Decider too


def _is_single_clip(v) -> bool:
    if isinstance(v, (str, os.PathLike, bytes, bytearray, dict)) or hasattr(v, "shape"):
        return True
    # (array, sampling_rate) is one clip; any other tuple is a collection of clips
    return isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], (int, np.integer)) and not isinstance(v[1], bool)


def _as_list(v) -> list:
    return [v] if _is_single_clip(v) else list(v)


class Choice:
    """A question whose answer is one of ``labels``.

    Build it with :meth:`from_descriptions` (zero-shot), :meth:`from_examples` (few-shot
    prototypes, recommended) or :meth:`from_pack`. Answer it with ``Decider.decide``.
    """

    def __init__(self, labels: Sequence[str], mode: str, *, descriptions: Mapping[str, list[str]] | None = None,
                 sums: np.ndarray | None = None, counts: np.ndarray | None = None, tau: float = cal.DEFAULT_TAU,
                 thresholds: Mapping[str, float] | None = None, min_similarity: float | None = None,
                 centering: bool = False, audio_mean: np.ndarray | None = None, name: str | None = None,
                 task: str = "custom", label_status: Mapping[str, str] | None = None,
                 model_id: str = MODEL_ID, model_revision: str | None = None,
                 fingerprints: Iterable[str] = (), text_prefix: str = CLASSIFICATION_PREFIX):
        if mode not in ("text", "proto"):
            raise ValueError("mode must be 'text' or 'proto'")
        labels = list(labels)
        if len(labels) < 2:
            raise ValueError("a Choice needs at least 2 options")
        if len(set(labels)) != len(labels):
            raise ValueError("option labels must be unique")
        self.labels = labels
        self.mode = mode
        self.descriptions = {k: [v] if isinstance(v, str) else list(v) for k, v in (descriptions or {}).items()}
        self.sums = None if sums is None else np.asarray(sums, np.float32)
        self.counts = None if counts is None else np.asarray(counts, np.int64)
        self.tau = float(tau)
        self.thresholds = dict(thresholds or cal.DEFAULT_THRESHOLDS)
        self.min_similarity = min_similarity
        self.centering = centering
        self.audio_mean = None if audio_mean is None else np.asarray(audio_mean, np.float32)
        self.name = name
        self.task = task
        self.label_status = dict(label_status or {})
        self.model_id = model_id
        self.model_revision = model_revision
        self.fingerprints = set(fingerprints)
        self.text_prefix = text_prefix
        self._text_cache: tuple | None = None   # (encoder id, phrase embeddings, owner index)
        if mode == "proto" and (self.sums is None or self.counts is None or len(self.sums) != len(labels)):
            raise ValueError("proto mode needs one prototype sum and count per label")
        if mode == "text":
            missing = [l for l in labels if not self.descriptions.get(l)]
            if missing:
                raise ValueError(f"text mode needs at least one description per option; missing: {missing}")

    def __repr__(self) -> str:
        return f"Choice(name={self.name!r}, mode={self.mode!r}, options={len(self.labels)})"

    # ── constructors ──
    @classmethod
    def from_descriptions(cls, descriptions: Mapping[str, str | Sequence[str] | None], *, centering: bool = False,
                          name: str | None = None, prefix: str = CLASSIFICATION_PREFIX) -> "Choice":
        """Zero-shot question (the default mode). Give 1-5 descriptive phrasings per option,
        e.g. ``"A caller reporting a lost or stolen card"`` rather than ``"lost card"``.
        Phrasings are embedded with the classification prefix and averaged per option.

        ``centering=True`` subtracts each modality's mean before comparing; it needs
        :meth:`adapt` to be called with some unlabeled clips from your audio first.
        """
        global _HINT_SHOWN
        desc = {}
        for label, d in descriptions.items():
            if d is None:
                d = [label.replace("_", " ")]
            desc[label] = [d] if isinstance(d, str) else list(d)
        if not _HINT_SHOWN:
            _HINT_SHOWN = True
            warnings.warn("Using zero-shot text anchors. For higher accuracy, build prototypes from ~10 labeled "
                          "clips per option: Choice.from_examples(...)", ZeroShotHint, stacklevel=2)
        return cls(list(desc), "text", descriptions=desc, centering=centering, name=name, text_prefix=prefix)

    @classmethod
    def from_examples(cls, examples: Mapping[str, Sequence[AudioInput]], *, encoder=None, name: str | None = None,
                      batch_size: int = 8, max_seconds: float | None = None,
                      descriptions: Mapping[str, Sequence[str]] | None = None) -> "Choice":
        """Few-shot prototypes (recommended). ``examples`` maps each option to its labeled clips
        (paths, bytes, arrays...). ~10 clips per option from your own audio, from varied speakers.
        No training happens: each option's clip embeddings are averaged into one prototype.
        """
        enc = _encoder_of(encoder)
        labels = list(examples)
        sums, counts, prints = cls._embed_examples(enc, examples, labels, batch_size, max_seconds)
        return cls(labels, "proto", sums=sums, counts=counts, name=name, descriptions=descriptions,
                   model_id=enc.model_id, model_revision=enc.revision, fingerprints=prints)

    @classmethod
    def from_embeddings(cls, embeddings: Mapping[str, np.ndarray], *, name: str | None = None,
                        model_id: str = MODEL_ID, model_revision: str | None = None) -> "Choice":
        """Prototypes from precomputed audio embeddings: ``{label: (n, 768) array}``."""
        labels = list(embeddings)
        sums = np.stack([_l2n(np.atleast_2d(np.asarray(embeddings[l], np.float32))).sum(0) for l in labels])
        counts = np.array([np.atleast_2d(embeddings[l]).shape[0] for l in labels])
        return cls(labels, "proto", sums=sums, counts=counts, name=name, model_id=model_id,
                   model_revision=model_revision)

    @classmethod
    def from_pack(cls, pack: str, *, languages: Sequence[str] | None = None, warn_untested: bool = True) -> "Choice":
        """Load a saved question: a built-in name (``"language-id"``), a local folder, or a
        Hugging Face Hub repo id. For language packs, only tested languages load by default;
        pass ``languages=[...]`` to pick others (untested ones emit a warning).
        """
        from .packs import load_pack

        return load_pack(pack, languages=languages, warn_untested=warn_untested)

    # ── editing ──
    @staticmethod
    def _embed_examples(enc, examples, labels, batch_size, max_seconds):
        sums, counts, prints = [], [], set()
        for label in labels:
            clips = [load_audio(a, max_seconds=max_seconds) for a in _as_list(examples[label])]
            if not clips:
                raise ValueError(f"option '{label}' has no clips")
            prints.update(fingerprint(c) for c in clips)
            emb = enc.embed_audio(clips, batch_size=batch_size)
            sums.append(emb.sum(0))
            counts.append(len(clips))
        return np.stack(sums).astype(np.float32), np.array(counts), prints

    def _require_proto(self, what: str):
        if self.mode != "proto":
            raise VoiceDecideError(f"{what} needs a prototype question; build one with Choice.from_examples(...)")

    def add_examples(self, examples: Mapping[str, Sequence[AudioInput]], *, encoder=None, batch_size: int = 8,
                     max_seconds: float | None = None) -> "Choice":
        """Add clips to existing options or create new options. Old clips are not re-embedded."""
        self._require_proto("add_examples")
        enc = _encoder_of(encoder)
        labels = list(examples)
        sums, counts, prints = self._embed_examples(enc, examples, labels, batch_size, max_seconds)
        new = []
        for label, s, c in zip(labels, sums, counts):
            if label in self.labels:
                i = self.labels.index(label)
                self.sums[i] += s
                self.counts[i] += c
            else:
                self.labels.append(label)
                self.sums = np.vstack([self.sums, s[None]])
                self.counts = np.append(self.counts, c)
                new.append(label)
        self.fingerprints |= prints
        if self.task == "language-id" and new:
            for label in new:
                self.label_status[label] = "untested"
            warnings.warn(untested_message(new), UntestedLanguageWarning, stacklevel=2)
        return self

    def remove_label(self, label: str) -> "Choice":
        if label not in self.labels:
            raise UnknownLabelError(f"'{label}' is not an option of this question")
        if len(self.labels) <= 2:
            raise ValueError("a Choice needs at least 2 options")
        i = self.labels.index(label)
        self.labels.pop(i)
        if self.mode == "proto":
            self.sums = np.delete(self.sums, i, axis=0)
            self.counts = np.delete(self.counts, i)
        self.descriptions.pop(label, None)
        self.label_status.pop(label, None)
        self._text_cache = None
        return self

    def subset(self, labels: Sequence[str]) -> "Choice":
        """A copy restricted to ``labels`` (in that order)."""
        unknown = [l for l in labels if l not in self.labels]
        if unknown:
            raise UnknownLabelError(f"not options of this question: {unknown}")
        new = copy.deepcopy(self)
        idx = [self.labels.index(l) for l in labels]
        new.labels = list(labels)
        if new.mode == "proto":
            new.sums, new.counts = self.sums[idx].copy(), self.counts[idx].copy()
        new.descriptions = {l: v for l, v in self.descriptions.items() if l in labels}
        new.label_status = {l: v for l, v in self.label_status.items() if l in labels}
        new._text_cache = None
        if len(new.labels) < 2:
            raise ValueError("a Choice needs at least 2 options")
        return new

    def adapt(self, unlabeled_audio: Sequence[AudioInput], *, encoder=None, batch_size: int = 8) -> "Choice":
        """Set the audio mean used by centering from unlabeled clips of your own audio (no labels needed)."""
        enc = _encoder_of(encoder)
        clips = [load_audio(a) for a in _as_list(unlabeled_audio)]
        if len(clips) < 10:
            warnings.warn(f"adapt() got {len(clips)} clips; 50+ gives a more stable mean", stacklevel=2)
        self.audio_mean = enc.embed_audio(clips, batch_size=batch_size).mean(0)
        self.centering = True
        return self

    # ── scoring ──
    def _anchors(self, enc: EncoderLike) -> np.ndarray:
        if self.mode == "proto":
            return _l2n(self.sums / self.counts[:, None])
        if self._text_cache is None or self._text_cache[0] != id(enc):
            phrases = [p for l in self.labels for p in self.descriptions[l]]
            owner = np.array([i for i, l in enumerate(self.labels) for _ in self.descriptions[l]])
            self._text_cache = (id(enc), enc.embed_text(phrases, prefix=self.text_prefix), owner)
        _, P, owner = self._text_cache
        if self.centering:
            P = _l2n(P - P.mean(0))
        return _l2n(np.stack([P[owner == i].mean(0) for i in range(len(self.labels))]))

    def _prepare_audio(self, emb: np.ndarray) -> np.ndarray:
        if self.mode == "text" and self.centering:
            if self.audio_mean is None:
                raise VoiceDecideError("centering=True needs an audio mean: call question.adapt(unlabeled_clips) first")
            return _l2n(emb - self.audio_mean)
        return emb

    def scores(self, emb: np.ndarray, enc: EncoderLike) -> np.ndarray:
        """Cosine similarity of (n, 768) audio embeddings with every option -> (n, K)."""
        return self._prepare_audio(np.atleast_2d(emb)) @ self._anchors(enc).T

    def answers_from_scores(self, scores: np.ndarray) -> list[Answer]:
        probs = cal.softmax(scores / self.tau)
        conf = cal.confidence(probs)
        out = []
        for s, p, c in zip(scores, probs, conf):
            order = np.argsort(p)[::-1]
            top = int(order[0])
            p_top = float(p[top])
            sim = float(s[top])
            oos = self.min_similarity is not None and sim < self.min_similarity
            action = "escalate" if oos else str(cal.actions(np.array([p_top]), self.thresholds)[0])
            out.append(Answer(choice=None if oos else self.labels[top],
                              probabilities={l: float(v) for l, v in zip(self.labels, p)},
                              confidence=float(c), margin=float(p_top - p[order[1]]), action=action,
                              mode=self.mode, similarity=sim, out_of_set=bool(oos)))
        return out

    # ── calibration ──
    def _val_scores(self, audio, labels, enc, batch_size, max_seconds):
        audio, labels = list(audio), list(labels)
        if len(audio) != len(labels):
            raise ValueError("audio and labels must have the same length")
        unknown = sorted(set(labels) - set(self.labels))
        if unknown:  # check before embedding anything
            raise UnknownLabelError(f"validation labels not in this question: {unknown}")
        clips = [load_audio(a, max_seconds=max_seconds) for a in audio]
        notes = []
        overlap = sum(fingerprint(c) in self.fingerprints for c in clips)
        if overlap:
            msg = (f"{overlap} of {len(clips)} validation clips were also used to build the prototypes; "
                   "metrics will look better than they are. Use different clips.")
            warnings.warn(msg, LeakageWarning, stacklevel=3)
            notes.append(msg)
        emb = enc.embed_audio(clips, batch_size=batch_size)
        return self._emb_scores(emb, labels, enc) + (notes,)

    def _emb_scores(self, emb, labels, enc):
        unknown = sorted(set(labels) - set(self.labels))
        if unknown:
            raise UnknownLabelError(f"validation labels not in this question: {unknown}")
        y = np.array([self.labels.index(l) for l in labels])
        return self.scores(np.asarray(emb, np.float32), enc), y

    def evaluate(self, audio: Sequence[AudioInput], labels: Sequence[str], *, encoder=None, batch_size: int = 8,
                 max_seconds: float | None = None) -> cal.FitReport:
        """Measure accuracy, macro-F1, ECE and action shares on labeled clips. Changes nothing."""
        enc = _encoder_of(encoder)
        S, y, notes = self._val_scores(audio, labels, enc, batch_size, max_seconds)
        return cal.build_report(S, y, self.labels, self.tau, self.thresholds, self.min_similarity, notes)

    def evaluate_embeddings(self, embeddings: np.ndarray, labels: Sequence[str], *, encoder=None) -> cal.FitReport:
        """Like evaluate(), from precomputed (n, 768) audio embeddings."""
        S, y = self._emb_scores(embeddings, list(labels), _encoder_of(encoder))
        return cal.build_report(S, y, self.labels, self.tau, self.thresholds, self.min_similarity)

    def fit(self, audio: Sequence[AudioInput], labels: Sequence[str], *, encoder=None,
            act_precision: float = 0.95, confirm_precision: float = 0.80, tau_grid=cal.TAU_GRID,
            batch_size: int = 8, max_seconds: float | None = None, out_of_set: bool = True) -> cal.FitReport:
        """Calibrate on held-out labeled clips (not the prototype clips):

        * tau: the grid value with the lowest negative log-likelihood
        * act threshold: lowest probability whose decisions are right >= ``act_precision`` of the time
        * confirm threshold: same with ``confirm_precision``
        * min_similarity (if ``out_of_set``): just below the lowest best-option cosine seen on the
          validation clips; audio scoring below it is flagged as matching none of the options.
          The floor only transfers to audio like the validation clips (same channel and style):
          calibrate on clips from your own pipeline.
        """
        enc = _encoder_of(encoder)
        S, y, notes = self._val_scores(audio, labels, enc, batch_size, max_seconds)
        return self._fit_scores(S, y, list(labels), enc, act_precision, confirm_precision, tau_grid, out_of_set, notes)

    def fit_embeddings(self, embeddings: np.ndarray, labels: Sequence[str], *, encoder=None,
                       act_precision: float = 0.95, confirm_precision: float = 0.80, tau_grid=cal.TAU_GRID,
                       out_of_set: bool = True) -> cal.FitReport:
        """Like fit(), from precomputed (n, 768) audio embeddings."""
        enc = _encoder_of(encoder)
        S, y = self._emb_scores(embeddings, list(labels), enc)
        return self._fit_scores(S, y, list(labels), enc, act_precision, confirm_precision, tau_grid, out_of_set, [])

    def _fit_scores(self, S, y, labels, enc, act_precision, confirm_precision, tau_grid, out_of_set, notes):
        self.tau = cal.pick_tau(S, y, tau_grid)
        probs = cal.softmax(S / self.tau)
        p_top, correct = probs.max(1), probs.argmax(1) == y
        act = cal.pick_threshold(p_top, correct, act_precision)
        conf = cal.pick_threshold(p_top, correct, confirm_precision)
        if act is None:
            act = 1.01
            notes.append(f"no threshold reached {act_precision:.0%} precision; nothing will auto-act "
                         "(add more or better prototype clips)")
        if conf is None:
            conf = act
            notes.append(f"no threshold reached {confirm_precision:.0%} precision; confirm is disabled")
        elif conf <= p_top.min():
            # No calibration clip scored lower, so there is no evidence about that range: keep the
            # default confirm floor there instead of escalating everything below the weakest clip.
            conf = min(conf, cal.DEFAULT_THRESHOLDS["confirm"])
        self.thresholds = {"act": act, "confirm": min(conf, act)}
        if self.tau in (min(tau_grid), max(tau_grid)):
            notes.append(f"tau={self.tau:g} is at the edge of the search grid; the calibration set may be too "
                         "small or too easy, so probabilities may be over-confident on new audio")
        if out_of_set:
            # every validation clip must pass; a percentile would flag the weakest in-set clips
            self.min_similarity = float(S.max(1).min() - OOS_MARGIN)
        for l in set(labels):
            if self.label_status.get(l) == "untested":
                self.label_status[l] = "user-evaluated"
        self.model_id, self.model_revision = enc.model_id, enc.revision or self.model_revision
        return cal.build_report(S, y, self.labels, self.tau, self.thresholds, self.min_similarity, notes)

    def save(self, path: str) -> str:
        from .packs import save_pack

        return save_pack(self, path)
