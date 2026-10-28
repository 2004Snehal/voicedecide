"""Save and load questions as packs: a folder with pack.json (+ prototype_sums.npy, audio_mean.npy).

A pack stores only vectors and settings, never audio.
"""

from __future__ import annotations

import json
import logging
import warnings
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import PackError, UntestedLanguageWarning
from .languages import TESTED, display_name, resolve, untested_message
from .model import CLASSIFICATION_PREFIX, MODEL_ID
from .questions import Choice

FORMAT_VERSION = 1
BUILTIN = {"language-id"}
log = logging.getLogger("voicedecide")


def save_pack(q: Choice, path: str, extra: dict | None = None) -> str:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT_VERSION, "name": q.name or out.name, "task": q.task, "mode": q.mode,
        "labels": q.labels, "descriptions": q.descriptions, "tau": q.tau, "thresholds": q.thresholds,
        "min_similarity": q.min_similarity, "centering": q.centering, "text_prefix": q.text_prefix,
        "model_id": q.model_id, "model_revision": q.model_revision, "dim": 768,
        "label_status": q.label_status, "fingerprints": sorted(q.fingerprints),
    }
    if q.mode == "proto":
        meta["counts"] = [int(c) for c in q.counts]
        np.save(out / "prototype_sums.npy", q.sums.astype(np.float32))
    if q.audio_mean is not None:
        np.save(out / "audio_mean.npy", q.audio_mean.astype(np.float32))
    meta.update(extra or {})
    (out / "pack.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return str(out)


def _locate(pack: str) -> Path:
    if pack in BUILTIN:
        p = Path(str(resources.files("voicedecide") / "packs" / pack))
        if not (p / "pack.json").exists():
            raise PackError(f"built-in pack '{pack}' is not included in this build of voicedecide "
                            "(build it with benchmarks/build_language_pack.py)")
        return p
    p = Path(pack).expanduser()
    if p.is_dir():
        if not (p / "pack.json").exists():
            raise PackError(f"{p} has no pack.json")
        return p
    if pack.count("/") == 1 and not p.exists():  # Hugging Face Hub repo id: "user/my-pack"
        from huggingface_hub import snapshot_download

        try:
            return Path(snapshot_download(pack, allow_patterns=["pack.json", "*.npy", "README.md"]))
        except Exception as e:
            raise PackError(f"pack '{pack}' is not a local folder, and downloading it as a Hugging Face Hub "
                            f"repo failed: {type(e).__name__}: {e}") from e
    raise PackError(f"pack '{pack}' not found (built-in names: {sorted(BUILTIN)}; or a folder or 'user/repo')")


def pack_meta(pack: str) -> dict:
    """Read a pack's pack.json without building the question."""
    return json.loads((_locate(pack) / "pack.json").read_text())


def load_pack(pack: str, *, languages: Sequence[str] | None = None, warn_untested: bool = True) -> Choice:
    root = _locate(pack)
    meta = json.loads((root / "pack.json").read_text())
    if meta.get("format", 1) > FORMAT_VERSION:
        raise PackError(f"pack format {meta['format']} is newer than this voicedecide supports; upgrade the library")
    sums = np.load(root / "prototype_sums.npy") if meta["mode"] == "proto" else None
    mean = np.load(root / "audio_mean.npy") if (root / "audio_mean.npy").exists() else None
    q = Choice(meta["labels"], meta["mode"], descriptions=meta.get("descriptions"), sums=sums,
               counts=meta.get("counts"), tau=meta["tau"], thresholds=meta["thresholds"],
               min_similarity=meta.get("min_similarity"), centering=meta.get("centering", False), audio_mean=mean,
               name=meta.get("name"), task=meta.get("task", "custom"), label_status=meta.get("label_status"),
               model_id=meta.get("model_id") or MODEL_ID, model_revision=meta.get("model_revision"),
               fingerprints=meta.get("fingerprints", ()),
               text_prefix=meta.get("text_prefix") or CLASSIFICATION_PREFIX)
    if q.task != "language-id":
        if languages is not None:
            raise PackError("languages=... only applies to language-id packs")
        return q
    return _select_languages(q, languages, warn_untested, pack)


def _status(q: Choice, code: str) -> str:
    # A label is only "tested" if the pack says so AND it is in the benchmarked set.
    s = q.label_status.get(code, "untested")
    return "tested" if s == "tested" and code in TESTED else s


def _select_languages(q: Choice, languages, warn_untested: bool, pack: str) -> Choice:
    if languages is None:
        chosen = [c for c in q.labels if _status(q, c) == "tested"]
        if len(chosen) < 2:  # e.g. a user-built language pack: nothing benchmarked, so load it whole
            chosen = list(q.labels)
        skipped = [c for c in q.labels if c not in chosen]
        if skipped:
            log.info("loaded %d tested languages; %d untested ones skipped (pass languages=[...] to use them)",
                     len(chosen), len(skipped))
    else:
        chosen = []
        for name in languages:
            code = resolve(name, q.labels, pack)
            if code not in chosen:
                chosen.append(code)
    untested = [c for c in chosen if _status(q, c) == "untested"]
    if untested and warn_untested:
        # stack: user -> Choice.from_pack -> load_pack -> here
        warnings.warn(untested_message(untested), UntestedLanguageWarning, stacklevel=4)
    return q.subset(chosen) if chosen != q.labels else q


def list_languages(pack: str = "language-id") -> list[dict]:
    """Languages in a pack with their status. Metrics are listed only for tested languages."""
    root = _locate(pack)
    meta = json.loads((root / "pack.json").read_text())
    metrics = meta.get("metrics", {}).get("per_language", {})
    q_status = meta.get("label_status", {})
    rows = []
    for code in meta["labels"]:
        status = "tested" if q_status.get(code) == "tested" and code in TESTED else q_status.get(code, "untested")
        row = {"code": code, "name": display_name(code), "status": status}
        if status == "tested" and code in metrics:
            row["test_accuracy"] = metrics[code]
        rows.append(row)
    return rows
