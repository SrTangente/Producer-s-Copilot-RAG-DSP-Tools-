import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import pytest
import sys
from pathlib import Path

# Ensure the repository root is on the Python path so ``tools`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools import analyze_audio


@pytest.fixture(autouse=True)
def patch_meter(monkeypatch):
    """Ensure pyln.Meter has a loudness_range method for tests."""
    class PatchedMeter(pyln.Meter):
        def loudness_range(self, *_args, **_kwargs):
            return 0.0
    monkeypatch.setattr(pyln, "Meter", PatchedMeter)


def test_bpm_detection(tmp_path):
    sr = 48000
    click = np.zeros(sr * 2, dtype=np.float32)
    for t in range(0, len(click), int(sr * 0.5)):
        click[t:t + 200] = 1.0
    path = tmp_path / "click.wav"
    sf.write(path, click, sr)
    metrics = analyze_audio(str(path))
    assert metrics["bpm"] == pytest.approx(120, abs=1)


def test_key_detection(tmp_path):
    sr = 48000
    C = librosa.tone(261.63, sr=sr, length=sr)
    E = librosa.tone(329.63, sr=sr, length=sr)
    G = librosa.tone(392.0, sr=sr, length=sr)
    chord = (C + E + G) / 3
    path = tmp_path / "chord.wav"
    sf.write(path, chord, sr)
    metrics = analyze_audio(str(path))
    assert metrics["key"] == "C"
    assert metrics["mode"] == "major"
    assert metrics["key_confidence"] > 0.8


def test_loudness_analysis(tmp_path):
    sr = 48000
    freq = 1000
    y = 0.5 * np.sin(2 * np.pi * freq * np.arange(sr) / sr)
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr)
    expected_lufs = pyln.Meter(sr).integrated_loudness(y)
    metrics = analyze_audio(str(path))
    assert metrics["lufs_i"] == pytest.approx(expected_lufs, abs=0.1)
    assert metrics["lra"] == pytest.approx(0.0, abs=0.01)
    assert metrics["true_peak_dbfs"] == pytest.approx(-6.02, abs=0.1)
