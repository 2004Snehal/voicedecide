import json
import re
import warnings

import numpy as np
import pytest
import soundfile as sf

from voicedecide import (Choice, Decider, ModelRevisionWarning, PackError, UnknownLanguageError,
                         UntestedLanguageWarning, cli, list_languages)
from voicedecide.audio import SR
from voicedecide.languages import TESTED, resolve, untested_message

from .conftest import FREQS, tone

# Map fake tone classes onto language codes: two tested, two untested.
LANGS = ["hi_in", "ta_in", "ur_pk", "gu_in"]
STATUS = {"hi_in": "tested", "ta_in": "tested", "ur_pk": "untested", "gu_in": "untested"}


@pytest.fixture
def lang_pack(tmp_path, fake):
    d = Decider(encoder=fake)
    q = Choice.from_examples({c: [(tone(k, seed=10 * k + i), SR) for i in range(5)] for k, c in enumerate(LANGS)},
                             encoder=d, name="language-id")
    q.task = "language-id"
    q.label_status = dict(STATUS)
    path = tmp_path / "language-id"
    from voicedecide.packs import save_pack

    save_pack(q, str(path), extra={"metrics": {"per_language": {"hi_in": 0.9, "ta_in": 0.8, "ur_pk": 0.4}}})
    return str(path)


def test_roundtrip_gives_same_decisions(tmp_path, fake, examples, val):
    d = Decider(encoder=fake)
    q = Choice.from_examples(examples, encoder=d)
    q.fit(*val, encoder=d)
    q2 = Choice.from_pack(q.save(str(tmp_path / "p")))
    assert q2.labels == q.labels and q2.tau == q.tau and q2.thresholds == q.thresholds
    assert q2.min_similarity == q.min_similarity and q2.fingerprints == q.fingerprints
    for k in range(3):
        a, b = (d.decide((tone(k, seed=300 + k), SR), {"q": x})["q"] for x in (q, q2))
        assert a.choice == b.choice and abs(a.probability - b.probability) < 1e-6


def test_text_pack_roundtrip(tmp_path, fake):
    q = Choice.from_descriptions({"c0": ["tone 0"], "c1": ["tone 1"]}, name="t")
    q2 = Choice.from_pack(q.save(str(tmp_path / "t")))
    assert q2.mode == "text" and q2.descriptions == q.descriptions


def test_default_loads_only_tested(lang_pack):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = Choice.from_pack(lang_pack)
    assert q.labels == ["hi_in", "ta_in"]


def test_untested_warning_has_no_numbers(lang_pack):
    with pytest.warns(UntestedLanguageWarning) as rec:
        q = Choice.from_pack(lang_pack, languages=["hindi", "ur", "Gujarati"])
    assert q.labels == ["hi_in", "ur_pk", "gu_in"]
    msg = str(rec[0].message)
    assert "Urdu" in msg and "Gujarati" in msg and "not been tested" in msg
    assert "Hindi" not in msg.split("Tested languages")[0]  # tested ones are not flagged
    body = msg.split("voicedecide eval")[0]
    assert not re.search(r"\d", body), f"warning must not show any figures: {body!r}"


def test_tested_status_requires_benchmark_set():
    # a pack cannot promote a language to tested on its own
    assert "ur_pk" not in TESTED and TESTED == {"en_us", "hi_in", "bn_in", "ta_in", "te_in", "mr_in"}


def test_warning_can_be_silenced(lang_pack):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Choice.from_pack(lang_pack, languages=["ur_pk", "hi_in"], warn_untested=False)


def test_unknown_language_suggests(lang_pack):
    with pytest.raises(UnknownLanguageError) as e:
        Choice.from_pack(lang_pack, languages=["hi_in", "urdo"])
    assert "Did you mean" in str(e.value) and "ur_pk" in str(e.value)


def test_resolve_aliases():
    avail = ["hi_in", "ta_in", "cmn_hans_cn"]
    assert resolve("Hindi", avail) == resolve("hi", avail) == resolve("HI_IN", avail) == "hi_in"
    assert resolve("chinese", avail) == "cmn_hans_cn"


def test_untested_message_fixed_text():
    assert not re.search(r"\d", untested_message(["ur_pk"]).split("voicedecide eval")[0])


def test_added_language_is_untested(lang_pack, fake):
    q = Choice.from_pack(lang_pack)
    with pytest.warns(UntestedLanguageWarning):
        q.add_examples({"ml_in": [(tone(4, seed=i), SR) for i in range(5)]}, encoder=fake)
    assert q.label_status["ml_in"] == "untested"
    q.fit([(tone(4, seed=60 + i), SR) for i in range(5)] + [(tone(0, seed=70 + i), SR) for i in range(5)],
          ["ml_in"] * 5 + ["hi_in"] * 5, encoder=fake)
    assert q.label_status["ml_in"] == "user-evaluated" and q.label_status["hi_in"] == "tested"


def test_list_languages_shows_metrics_only_for_tested(lang_pack):
    rows = {r["code"]: r for r in list_languages(lang_pack)}
    assert rows["hi_in"]["status"] == "tested" and "test_accuracy" in rows["hi_in"]
    assert rows["ur_pk"]["status"] == "untested" and "test_accuracy" not in rows["ur_pk"]


def test_languages_kwarg_only_for_language_packs(tmp_path, fake, examples):
    path = Choice.from_examples(examples, encoder=fake).save(str(tmp_path / "r"))
    with pytest.raises(PackError):
        Choice.from_pack(path, languages=["hi"])


def test_revision_mismatch_warns(tmp_path, fake, examples):
    q = Choice.from_examples(examples, encoder=fake)
    q.model_revision = "someotherrev"
    with pytest.warns(ModelRevisionWarning):
        Decider(encoder=fake).decide((tone(0), SR), {"q": q})


def test_missing_pack_errors(tmp_path):
    with pytest.raises(PackError):
        Choice.from_pack(str(tmp_path / "nope"))


def test_builtin_pack_absent_gives_clear_error():
    from voicedecide.packs import _locate

    try:
        _locate("language-id")
    except PackError as e:
        assert "build_language_pack" in str(e)


# ── CLI ──
def _write_dirs(root, n_per=5, seed0=0):
    for k in range(3):
        d = root / f"c{k}"
        d.mkdir(parents=True)
        for i in range(n_per):
            sf.write(d / f"{i}.wav", tone(k, seed=seed0 + 10 * k + i), SR)


def test_cli_build_run_eval(tmp_path, capsys):
    _write_dirs(tmp_path / "ex")
    _write_dirs(tmp_path / "val", n_per=8, seed0=500)
    cli.main(["build-pack", "--examples", str(tmp_path / "ex"), "--val", str(tmp_path / "val"),
              "--out", str(tmp_path / "pack")])
    assert (tmp_path / "pack" / "pack.json").exists()
    capsys.readouterr()

    files = [str(tmp_path / "val" / f"c{k}" / "0.wav") for k in range(3)]
    cli.main(["run", "--pack", f"route={tmp_path / 'pack'}", *files])
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert [l["answers"]["route"]["choice"] for l in lines] == ["c0", "c1", "c2"]

    csv = tmp_path / "labels.csv"
    csv.write_text("path,label\n" + "".join(f"{f},c{k}\n" for k, f in enumerate(files)))
    cli.main(["eval", "--pack", str(tmp_path / "pack"), "--json", str(csv)])
    assert json.loads(capsys.readouterr().out)["accuracy"] == 1.0


def test_cli_languages(lang_pack, capsys):
    cli.main(["languages", "--pack", lang_pack])
    out = capsys.readouterr().out
    assert "ur_pk" in out and "untested" in out and not re.search(r"\d+%", out)


def test_fake_tones_are_distinct():
    assert len(set(FREQS)) == len(FREQS) and np.all(np.diff(FREQS) > 50)


def test_language_pack_without_tested_languages_loads_whole(tmp_path, fake):
    q = Choice.from_examples({c: [(tone(k, seed=k), SR)] for k, c in enumerate(["xx_aa", "yy_bb", "zz_cc"])},
                             encoder=fake)
    q.task = "language-id"
    q.label_status = {c: "untested" for c in q.labels}
    path = q.save(str(tmp_path / "mine"))
    with pytest.warns(UntestedLanguageWarning):
        q2 = Choice.from_pack(path)
    assert q2.labels == q.labels


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["languages", "--pack", str(tmp_path / "missing")])
    assert str(e.value).startswith("error: ") and "not found" in str(e.value)
