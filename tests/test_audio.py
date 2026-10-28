import io
import warnings

import numpy as np
import pytest
import soundfile as sf

from voicedecide.audio import MAX_SECONDS, SR, fingerprint, load_audio
from voicedecide.errors import AudioTruncatedWarning

from .conftest import tone


def test_path_bytes_and_fileobj_agree(tmp_path):
    x = tone(0, sr=8000)
    p = tmp_path / "a.wav"
    sf.write(p, x, 8000)
    a = load_audio(str(p))
    b = load_audio(p.read_bytes())
    c = load_audio(io.BytesIO(p.read_bytes()))
    assert a.dtype == np.float32 and a.ndim == 1
    assert len(a) == SR  # 1 s at 8 kHz resampled to 16 kHz
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, c)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
def test_resampling_length(sr):
    assert abs(len(load_audio((tone(1, seconds=2, sr=sr), sr))) - 2 * SR) <= 1


def test_stereo_downmix_both_layouts():
    x = tone(0)
    frames_first = np.stack([x, x], axis=1)
    channels_first = np.stack([x, x], axis=0)
    np.testing.assert_allclose(load_audio((frames_first, SR)), x, atol=1e-6)
    np.testing.assert_allclose(load_audio((channels_first, SR)), x, atol=1e-6)


def test_int16_is_scaled():
    x = (tone(0) * 32767).astype(np.int16)
    y = load_audio((x, SR))
    assert np.abs(y).max() <= 1.0 and np.abs(y).max() > 0.3


def test_hf_dict_inputs(tmp_path):
    x = tone(2)
    np.testing.assert_allclose(load_audio({"array": x, "sampling_rate": SR}), x)
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV")
    assert len(load_audio({"bytes": buf.getvalue(), "path": None})) == len(x)


def test_array_without_rate_errors():
    with pytest.raises(ValueError, match="sampling rate"):
        load_audio(tone(0))


def test_empty_errors():
    with pytest.raises(ValueError, match="empty"):
        load_audio((np.zeros(0, np.float32), SR))


def test_max_seconds_and_context_limit():
    x = np.zeros(SR * 10, np.float32)
    assert len(load_audio((x, SR), max_seconds=3)) == 3 * SR
    long = np.zeros(SR * (MAX_SECONDS + 5), np.float32)
    with pytest.warns(AudioTruncatedWarning):
        assert len(load_audio((long, SR))) == MAX_SECONDS * SR
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_audio((long, SR), max_seconds=5)  # explicit crop: no warning


def test_fingerprint_stable():
    x = tone(0)
    assert fingerprint(x) == fingerprint(x.copy()) != fingerprint(tone(1))
