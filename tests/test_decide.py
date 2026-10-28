import json
import warnings

import numpy as np
import pytest

from voicedecide import Choice, Decider, OutOfSetError, VoiceDecideError, ZeroShotHint
from voicedecide import questions as vq
from voicedecide.audio import SR
from voicedecide.errors import LeakageWarning, UnknownLabelError

from .conftest import tone

DESC = {"c0": ["a tone 0 sound", "low tone 0"], "c1": "a tone 1 sound", "c2": ["tone 2"]}


@pytest.fixture
def d(fake):
    return Decider(encoder=fake)


def test_text_anchors_zero_shot(d):
    vq._HINT_SHOWN = False
    with pytest.warns(ZeroShotHint):
        q = Choice.from_descriptions(DESC)
    assert q.mode == "text"
    for k in range(3):
        a = d.decide((tone(k, seed=7), SR), {"q": q})["q"]
        assert a.choice == f"c{k}"
        assert abs(sum(a.probabilities.values()) - 1) < 1e-6
        assert 0 <= a.confidence <= 1 and 0 <= a.margin <= 1
        assert a.action in {"act", "confirm", "escalate"} and a.mode == "text"


def test_hint_only_once():
    vq._HINT_SHOWN = False
    with pytest.warns(ZeroShotHint):
        Choice.from_descriptions(DESC)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Choice.from_descriptions(DESC)


def test_prototypes_and_one_embedding_for_many_questions(d, fake, examples):
    q = Choice.from_examples(examples, encoder=d)
    assert q.mode == "proto" and list(q.counts) == [6, 6, 6]
    text = Choice.from_descriptions(DESC)
    before = fake.audio_calls
    res = d.decide((tone(2, seed=9), SR), {"proto": q, "text": text})
    assert fake.audio_calls - before == 1  # one audio embedding answers both questions
    assert res["proto"].choice == "c2" and res["text"].choice == "c2"
    assert set(res.latency_ms) == {"decode", "embed", "decide", "total"}
    json.loads(res.to_json())


def test_actions_follow_thresholds(d, examples):
    q = Choice.from_examples(examples, encoder=d)
    q.thresholds = {"act": 1.01, "confirm": 1.01}
    assert d.decide((tone(0, seed=3), SR), {"q": q})["q"].action == "escalate"
    q.thresholds = {"act": 0.0, "confirm": 0.0}
    assert d.decide((tone(0, seed=3), SR), {"q": q})["q"].action == "act"


def test_add_examples_is_incremental(d, fake, examples):
    q = Choice.from_examples({k: v[:3] for k, v in examples.items()}, encoder=d)
    full = Choice.from_examples(examples, encoder=d)
    before = fake.audio_calls
    q.add_examples({k: v[3:] for k, v in examples.items()}, encoder=d)
    assert fake.audio_calls - before == 9  # only the new clips were embedded
    np.testing.assert_allclose(q.sums, full.sums, atol=1e-5)
    q.add_examples({"c3": [(tone(3, seed=i), SR) for i in range(4)]}, encoder=d)
    assert q.labels[-1] == "c3"
    assert d.decide((tone(3, seed=50), SR), {"q": q})["q"].choice == "c3"


def test_remove_and_subset(d, examples):
    q = Choice.from_examples(examples, encoder=d)
    s = q.subset(["c2", "c0"])
    assert s.labels == ["c2", "c0"] and q.labels == ["c0", "c1", "c2"]
    q.remove_label("c1")
    assert q.labels == ["c0", "c2"] and len(q.sums) == 2
    with pytest.raises(UnknownLabelError):
        q.remove_label("nope")
    with pytest.raises(ValueError):
        q.remove_label("c0")  # would leave one option


def test_add_examples_needs_proto(d):
    with pytest.raises(VoiceDecideError, match="from_examples"):
        Choice.from_descriptions(DESC).add_examples({"c0": [(tone(0), SR)]}, encoder=d)


def test_fit_calibrates_and_reports(d, examples, val):
    q = Choice.from_examples(examples, encoder=d)
    report = q.fit(*val, encoder=d)
    assert report.n == 36 and report.accuracy > 0.9
    assert q.tau in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
    assert q.thresholds["confirm"] <= q.thresholds["act"]
    assert q.min_similarity is not None
    assert "accuracy" in str(report)
    ev = q.evaluate(*val, encoder=d)
    assert ev.accuracy == report.accuracy and ev.tau == q.tau


def test_fit_warns_on_leakage(d, examples):
    q = Choice.from_examples(examples, encoder=d)
    audio = [c for v in examples.values() for c in v]
    labels = [k for k, v in examples.items() for _ in v]
    with pytest.warns(LeakageWarning):
        q.fit(audio, labels, encoder=d)


def test_fit_rejects_unknown_labels(d, examples, val):
    q = Choice.from_examples(examples, encoder=d)
    with pytest.raises(UnknownLabelError):
        q.fit(val[0][:2], ["c0", "zzz"], encoder=d)


def test_out_of_set_flag_and_strict(fake, examples, val):
    d = Decider(encoder=fake)
    q = Choice.from_examples(examples, encoder=d)
    q.fit(*val, encoder=d)
    odd = (tone(0, freq=1500.0, seed=4), SR)  # not one of the trained tones
    a = d.decide(odd, {"q": q})["q"]
    assert a.out_of_set and a.choice is None and a.action == "escalate"
    assert not d.decide((tone(1, seed=77), SR), {"q": q})["q"].out_of_set
    with pytest.raises(OutOfSetError):
        Decider(encoder=fake, strict=True).decide(odd, {"q": q})


def test_centering_needs_adapt(d):
    q = Choice.from_descriptions(DESC, centering=True)
    with pytest.raises(VoiceDecideError, match="adapt"):
        d.decide((tone(0), SR), {"q": q})
    q.adapt([(tone(k, seed=s), SR) for k in range(3) for s in range(5)], encoder=d)
    assert d.decide((tone(1, seed=8), SR), {"q": q})["q"].choice == "c1"


def test_decide_batch_matches_decide(d, examples):
    q = Choice.from_examples(examples, encoder=d)
    clips = [(tone(k, seed=200 + k), SR) for k in range(3)] * 3
    batch = d.decide_batch(clips, {"q": q}, batch_size=4)
    single = [d.decide(c, {"q": q}) for c in clips]
    assert [b["q"].choice for b in batch] == [s["q"].choice for s in single]
    for b, s in zip(batch, single):
        assert abs(b["q"].probability - s["q"].probability) < 1e-5


def test_bad_inputs(d):
    with pytest.raises(ValueError):
        d.decide((tone(0), SR), {})
    with pytest.raises(TypeError):
        d.decide((tone(0), SR), {"q": "not a choice"})
    with pytest.raises(ValueError):
        Choice.from_descriptions({"only": "one"})


def test_fit_never_flags_its_own_validation_clips(d, examples, val):
    q = Choice.from_examples(examples, encoder=d)
    report = q.fit(*val, encoder=d)
    assert report.out_of_set_share == 0.0
    assert not any(d.decide(a, {"q": q})["q"].out_of_set for a in val[0])


def test_examples_accept_tuples_of_paths_and_pathlib(tmp_path, d):
    import soundfile as sf

    paths = []
    for k in range(2):
        for i in range(3):
            p = tmp_path / f"{k}_{i}.wav"
            sf.write(p, tone(k, seed=40 + 10 * k + i), SR)
            paths.append(p)
    q = Choice.from_examples({"c0": tuple(str(p) for p in paths[:3]), "c1": paths[3:]}, encoder=d)
    assert list(q.counts) == [3, 3]
    q1 = Choice.from_examples({"c0": paths[0], "c1": (tone(1, seed=1), SR)}, encoder=d)  # single clips
    assert list(q1.counts) == [1, 1]
    res = d.decide(paths[4], {"q": q})
    assert res["q"].choice == "c1" and res.source == str(paths[4])


def test_string_descriptions_are_not_split(d, examples):
    q = Choice.from_examples(examples, encoder=d, descriptions={"c0": "a low tone"})
    assert q.descriptions["c0"] == ["a low tone"]


def test_perfect_calibration_keeps_a_confirm_zone(d, examples, val):
    q = Choice.from_examples(examples, encoder=d)
    report = q.fit(*val, encoder=d)
    assert report.accuracy == 1.0
    assert q.thresholds["confirm"] <= 0.5 < q.thresholds["act"]
