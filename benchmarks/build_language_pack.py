"""Build the built-in `language-id` pack from FLEURS (run on a GPU machine, e.g. Kaggle T4).

    pip install voicedecide pyarrow pandas
    python benchmarks/build_language_pack.py --out voicedecide/packs/language-id

What it does, per language:
  * prototypes: --proto clips from the FLEURS *validation* split (one recording per sentence)
  * calibration: --calib further validation clips (different sentences), used by fit()
  * benchmark: --test clips from the FLEURS *test* split, only for the tested languages
    (voicedecide.languages.TESTED); those figures go into pack.json["metrics"]

Downloads ~160 MB (validation) per language plus ~300 MB (test) per tested language; each file is
deleted after use so disk use stays small. Untested languages get no figures in the pack.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from voicedecide import Choice, Decider
from voicedecide.audio import load_audio
from voicedecide.languages import LANGUAGES, TESTED, display_name
from voicedecide.packs import save_pack


def fleurs_clips(lang: str, split: str, n: int, skip_ids: set[int], seed: int, tmp: str):
    path = hf_hub_download("google/fleurs", f"parquet-data/{lang}/{split}-00000-of-00001.parquet",
                           repo_type="dataset", local_dir=tmp)
    t = pq.read_table(path, columns=["id", "audio"]).to_pandas().drop_duplicates("id")
    t = t[~t.id.isin(skip_ids)]
    t = t.sample(n=min(n, len(t)), random_state=seed)
    clips = [load_audio(a) for a in t.audio]
    ids = set(int(i) for i in t.id)
    shutil.rmtree(tmp, ignore_errors=True)
    return clips, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", default=",".join(LANGUAGES))
    ap.add_argument("--proto", type=int, default=50)
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    langs = args.languages.split(",")
    unknown = [l for l in langs if l not in LANGUAGES]
    assert not unknown, f"unknown language codes: {unknown}"

    d = Decider()
    enc = d.encoder.load()
    print("model", enc.model_id, "revision", enc.revision, "device", enc.device, flush=True)

    proto_emb, calib_emb, calib_y, test_emb, test_y = {}, [], [], [], []
    t0 = time.time()
    for lang in langs:
        tmp = tempfile.mkdtemp()
        clips, ids = fleurs_clips(lang, "validation", args.proto + args.calib, set(), args.seed, tmp)
        emb = enc.embed_audio(clips, batch_size=8)
        proto_emb[lang] = emb[: args.proto]
        calib_emb.append(emb[args.proto:])
        calib_y += [lang] * len(emb[args.proto:])
        if lang in TESTED:
            clips, _ = fleurs_clips(lang, "test", args.test, set(), args.seed, tempfile.mkdtemp())
            test_emb.append(enc.embed_audio(clips, batch_size=8))
            test_y += [lang] * len(clips)
        print(f"{lang:12s} {display_name(lang):18s} proto={len(proto_emb[lang])} "
              f"calib={len(emb) - len(proto_emb[lang])}  {time.time() - t0:.0f}s", flush=True)

    q = Choice.from_embeddings(proto_emb, name="language-id", model_id=enc.model_id, model_revision=enc.revision)
    q.task = "language-id"
    q.label_status = {l: ("tested" if l in TESTED else "untested") for l in langs}
    fit = q.fit_embeddings(np.concatenate(calib_emb), calib_y, encoder=enc)
    # fit() promotes untested labels to "user-evaluated"; the built-in pack keeps the strict policy
    q.label_status = {l: ("tested" if l in TESTED else "untested") for l in langs}
    # A floor calibrated on FLEURS read speech flags correctly identified phone calls (8 kHz) as out
    # of set, so the built-in pack ships without one; users set it with lang.fit() on their own audio.
    q.min_similarity = None
    print("\ncalibration (validation split, all languages):\n" + str(fit), flush=True)

    metrics = {}
    if test_emb:
        # Benchmark tested languages against the full pack, so confusions with untested languages count.
        rep = q.evaluate_embeddings(np.concatenate(test_emb), test_y, encoder=enc)
        metrics = {"split": "FLEURS test", "n": rep.n, "per_language": rep.per_label_accuracy,
                   "accuracy_tested": rep.accuracy}
        print("\nbenchmark (FLEURS test, tested languages only):\n" + str(rep), flush=True)

    extra = {"metrics": metrics,
             "sources": [{"dataset": "google/fleurs", "license": "CC-BY-4.0",
                          "splits": {"prototypes": "validation", "calibration": "validation", "benchmark": "test"}}]}
    print("\nsaved", save_pack(q, args.out, extra=extra))
    print(json.dumps({"tested": sorted(TESTED), "untested": [l for l in langs if l not in TESTED]}, indent=1))


if __name__ == "__main__":
    main()
