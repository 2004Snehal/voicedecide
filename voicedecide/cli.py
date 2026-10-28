"""Command line: voicedecide download | languages | build-pack | run | eval"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".aiff", ".aif"}


def _labeled_dir(root: str) -> dict[str, list[str]]:
    """root/<label>/*.wav -> {label: [paths]}"""
    base = Path(root)
    if not base.is_dir():
        sys.exit(f"error: {root} is not a folder")
    out = {d.name: sorted(str(f) for f in d.iterdir() if f.suffix.lower() in AUDIO_EXT)
           for d in sorted(base.iterdir()) if d.is_dir()}
    out = {k: v for k, v in out.items() if v}
    if len(out) < 2:
        sys.exit(f"error: {root} needs at least 2 sub-folders of audio files, one per option")
    return out


def _labeled_csv(path: str) -> tuple[list[str], list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"path", "label"} <= set(rows[0]):
        sys.exit(f"error: {path} needs a header with 'path' and 'label' columns")
    base = Path(path).parent
    paths = [r["path"] if Path(r["path"]).is_absolute() else str(base / r["path"]) for r in rows]
    return paths, [r["label"] for r in rows]


def _flatten(d: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    paths, labels = [], []
    for label, files in d.items():
        paths += files
        labels += [label] * len(files)
    return paths, labels


def _questions(specs: list[str], languages: str | None):
    """--pack values ('path' or 'name=path') -> {question name: Choice}. --languages applies to language packs."""
    from .packs import pack_meta
    from .questions import Choice

    qs = {}
    for spec in specs:
        if "=" in spec:
            name, src = spec.split("=", 1)
        else:
            name, src = Path(spec.rstrip("/")).name, spec
        is_lang = pack_meta(src).get("task") == "language-id"
        qs[name] = Choice.from_pack(src, languages=languages.split(",") if languages and is_lang else None)
    return qs


def cmd_download(a):
    from huggingface_hub import snapshot_download

    from .model import MODEL_ID

    print(snapshot_download(a.model or MODEL_ID, revision=a.revision))


def cmd_languages(a):
    from .packs import list_languages

    rows = list_languages(a.pack)
    print(f"{'code':13s} {'name':18s} status")
    for r in rows:
        print(f"{r['code']:13s} {r['name']:18s} {r['status']}")


def cmd_build_pack(a):
    from .decide import Decider
    from .questions import Choice

    d = Decider(revision=a.revision)
    examples = _labeled_dir(a.examples)
    print(f"building prototypes: {len(examples)} options, {sum(map(len, examples.values()))} clips", file=sys.stderr)
    q = Choice.from_examples(examples, encoder=d, name=a.name or Path(a.out).name, max_seconds=a.max_seconds)
    if a.val:
        report = q.fit(*_flatten(_labeled_dir(a.val)), encoder=d, max_seconds=a.max_seconds)
        print(report, file=sys.stderr)
    else:
        print("no --val folder: keeping default tau and thresholds (run 'voicedecide eval' before going live)",
              file=sys.stderr)
    print(q.save(a.out))


def cmd_run(a):
    from .decide import Decider

    qs = _questions(a.pack, a.languages)
    d = Decider(revision=a.revision)
    for res in d.decide_batch(a.audio, qs, batch_size=a.batch_size, max_seconds=a.max_seconds):
        print(res.to_json())


def cmd_eval(a):
    from .decide import Decider

    qs = _questions([a.pack], a.languages)
    (name, q), = qs.items()
    d = Decider(revision=a.revision)
    paths, labels = _labeled_csv(a.csv)
    report = q.evaluate(paths, labels, encoder=d, max_seconds=a.max_seconds)
    print(json.dumps(report.to_dict(), indent=2) if a.json else report)


def main(argv=None):
    p = argparse.ArgumentParser(prog="voicedecide", description="Training-free audio decisions on EmbeddingGemma 2")
    p.add_argument("--revision", help="pin a model revision (commit hash)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download the model into the Hugging Face cache")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("languages", help="list languages in a language pack and whether they are tested")
    s.add_argument("--pack", default="language-id")
    s.set_defaults(fn=cmd_languages)

    s = sub.add_parser("build-pack", help="build a prototype pack from labeled folders (examples/<label>/*.wav)")
    s.add_argument("--examples", required=True, help="folder with one sub-folder of clips per option")
    s.add_argument("--val", help="held-out folder, same layout; calibrates tau and thresholds")
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--max-seconds", type=float)
    s.set_defaults(fn=cmd_build_pack)

    s = sub.add_parser("run", help="decide on audio files; prints one JSON line per file")
    s.add_argument("--pack", action="append", required=True, help="pack (repeatable); 'name=path' sets the question name")
    s.add_argument("--languages", help="comma-separated codes/names for language-id packs")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--max-seconds", type=float)
    s.add_argument("audio", nargs="+")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval", help="measure a pack on labeled clips (CSV with path,label columns)")
    s.add_argument("--pack", required=True)
    s.add_argument("--languages")
    s.add_argument("--max-seconds", type=float)
    s.add_argument("--json", action="store_true")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    from .errors import VoiceDecideError

    try:
        a.fn(a)
    except (VoiceDecideError, FileNotFoundError, ValueError) as e:  # user-facing problems: no traceback
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
