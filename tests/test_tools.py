import numpy as np
import pyloudnorm as pyln
import pytest

from app.tools import AudioLoadError, analyze_audio, loudness_range
from tests.conftest import SR, drum_loop


def tone(freq, seconds=2.0, sr=SR):
    return np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)


def test_bpm_detection(make_wav):
    sr = 48000
    click = np.zeros(sr * 6, dtype=np.float32)
    for t in range(0, len(click), int(sr * 0.5)):
        click[t:t + 200] = 1.0
    metrics = analyze_audio(make_wav(click, sr))
    assert metrics["bpm"] == pytest.approx(120, abs=1.5)


def test_key_detection_major_chord(make_wav):
    chord = sum(tone(f) for f in (261.63, 329.63, 392.0)) / 3
    metrics = analyze_audio(make_wav(chord))
    assert (metrics["key"], metrics["mode"]) == ("C", "major")
    assert metrics["key_confidence"] > 0.5


def test_key_detection_minor_chord(make_wav):
    chord = sum(tone(f) for f in (220.0, 261.63, 329.63)) / 3
    metrics = analyze_audio(make_wav(chord))
    assert (metrics["key"], metrics["mode"]) == ("A", "minor")


def test_loudness_and_peak_of_sine(make_wav):
    sr = 48000
    y = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(sr * 5) / sr)
    expected_lufs = pyln.Meter(sr).integrated_loudness(y)
    metrics = analyze_audio(make_wav(y, sr))
    assert metrics["lufs_i"] == pytest.approx(expected_lufs, abs=0.1)
    assert metrics["lra"] == pytest.approx(0.0, abs=0.1)
    assert metrics["true_peak_dbfs"] == pytest.approx(-6.02, abs=0.1)
    assert metrics["clipped_samples"] == 0


def test_loudness_range_reflects_level_changes():
    sr = 16000
    t = np.arange(sr * 10) / sr
    tone = np.sin(2 * np.pi * 1000 * t)
    y = np.concatenate([0.05 * tone, 0.5 * tone])  # 20 dB jump
    assert loudness_range(y, sr) > 15


def test_true_peak_catches_inter_sample_overs(make_wav):
    # A sine at fs/4 with a 45 degree phase offset: every sample lands at 0.707
    # of the real peak, so the sample peak under-reads by 3 dB.
    n = np.arange(SR)
    y = 0.9 * np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4)
    m = analyze_audio(make_wav(y))
    assert m["sample_peak_dbfs"] == pytest.approx(20 * np.log10(0.9 * 0.7071), abs=0.1)
    assert m["true_peak_dbfs"] == pytest.approx(20 * np.log10(0.9), abs=0.3)


def test_clipping_detected(make_wav):
    y = np.clip(1.5 * np.sin(2 * np.pi * 100 * np.arange(SR) / SR), -1, 1)
    assert analyze_audio(make_wav(y))["clipped_samples"] > 100


def test_tonal_balance_low_tone_lands_in_low_band(make_wav):
    y = 0.5 * np.sin(2 * np.pi * 100 * np.arange(SR * 2) / SR)
    bands = analyze_audio(make_wav(y))["band_level_db"]
    assert bands["low"] == pytest.approx(0.0, abs=0.1)
    assert max(bands["sub"], bands["low_mid"], bands["high"]) < -30


def test_stereo_metrics(make_wav):
    mono = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR)
    m = analyze_audio(make_wav(np.stack([mono, mono], axis=1)))
    assert m["stereo_width"] == pytest.approx(0.0, abs=1e-6)
    assert m["lr_correlation"] == pytest.approx(1.0, abs=1e-6)

    inverted = analyze_audio(make_wav(np.stack([mono, -mono], axis=1)))
    assert inverted["lr_correlation"] == pytest.approx(-1.0, abs=1e-6)


def test_full_mix_returns_all_keys(make_wav):
    m = analyze_audio(make_wav(drum_loop()))
    for key in ("bpm", "key", "lufs_i", "lra", "true_peak_dbfs", "plr_db",
                "crest_factor_db", "band_level_db", "stereo_width", "duration_s"):
        assert key in m
    assert m["channels"] == 2
    assert m["bpm"] == pytest.approx(120, abs=3)


def test_garbage_file_raises(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(AudioLoadError):
        analyze_audio(str(bad))


def test_block_boundaries_do_not_change_results(make_wav, monkeypatch):
    path = make_wav(drum_loop(seconds=12))
    whole = analyze_audio(path)
    monkeypatch.setattr("app.tools.BLOCK_FRAMES", 4097)
    split = analyze_audio(path)
    for key in ("lufs_i", "lra", "true_peak_dbfs", "stereo_width", "low_end_width", "bpm"):
        assert split[key] == pytest.approx(whole[key], abs=0.02), key
    assert split["clipped_samples"] == whole["clipped_samples"]


def test_stereo_loudness_matches_reference_meter(make_wav):
    y = drum_loop(seconds=10).astype(np.float64)
    expected = pyln.Meter(SR).integrated_loudness(y)
    assert analyze_audio(make_wav(y))["lufs_i"] == pytest.approx(expected, abs=0.05)


def test_silence_and_very_short_clips_do_not_crash(make_wav):
    silent = analyze_audio(make_wav(np.zeros(SR)))
    assert silent["lufs_i"] == -70.0 and silent["bpm"] == 0.0
    assert analyze_audio(make_wav(0.5 * tone(440, 0.05)))["duration_s"] == pytest.approx(0.05, abs=0.001)
