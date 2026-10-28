"""Real-model tests (slow, ~2.5 GB model download on first run): pytest -m slow"""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real(request):
    import numpy as np

    from voicedecide.model import Encoder

    return Encoder().load(), np


@pytest.fixture(scope="module")
def minds():
    pq = pytest.importorskip("pyarrow.parquet")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("PolyAI/minds14", "en-US/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(path, columns=["audio", "intent_class"]).to_pandas()
    names = {4: "balance", 6: "card_issues", 11: "joint_account"}
    t = t[t.intent_class.isin(names)]
    return [(a, names[i]) for a, i in zip(t.audio, t.intent_class)]


def test_embeddings_are_finite_768(real, minds):
    enc, np = real
    from voicedecide.audio import load_audio

    e = enc.embed_audio([load_audio(minds[0][0])])
    assert e.shape == (1, 768) and np.isfinite(e).all() and abs(np.linalg.norm(e) - 1) < 1e-4
    t = enc.embed_text(["A caller asking for their balance"])
    assert t.shape == (1, 768)


def test_zero_shot_and_prototypes_beat_chance(real, minds, monkeypatch):
    enc, np = real
    from voicedecide import Choice, Decider

    d = Decider(encoder=enc)
    by = {}
    for a, l in minds:
        by.setdefault(l, []).append(a)
    test = [(a, l) for l, v in by.items() for a in v[5:15]]
    zs = Choice.from_descriptions({
        "balance": ["A caller asking how much money is in their account"],
        "card_issues": ["A caller reporting a lost or stolen card"],
        "joint_account": ["A caller wanting to open a joint account with their partner"],
    })
    r = zs.evaluate([a for a, _ in test], [l for _, l in test], encoder=d)
    assert r.accuracy > 0.5  # chance is 1/3
    proto = Choice.from_examples({l: v[:5] for l, v in by.items()}, encoder=d)
    r = proto.evaluate([a for a, _ in test], [l for _, l in test], encoder=d)
    assert r.accuracy > 0.5
    res = d.decide(test[0][0], {"zs": zs, "proto": proto})
    assert res["zs"].choice in by and res.latency_ms["embed"] > 0
