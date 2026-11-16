import numpy as np
import pytest
import soundfile as sf

SR = 44100


def _write(path, y, sr=SR):
    sf.write(path, y, sr)
    return str(path)


@pytest.fixture
def make_wav(tmp_path):
    """Write a numpy array to a wav file and return its path."""
    counter = {"n": 0}

    def _make(y, sr=SR, name=None):
        counter["n"] += 1
        return _write(tmp_path / (name or f"clip{counter['n']}.wav"), y, sr)

    return _make


def drum_loop(seconds=8.0, bpm=120, sr=SR, gain=0.5, seed=0):
    """Kick on every beat plus a hat on the off-beat and a bass tone, stereo."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = 0.15 * np.sin(2 * np.pi * 55 * t)  # bass drone
    beat = int(sr * 60 / bpm)
    kick_len = int(0.12 * sr)
    kt = np.arange(kick_len) / sr
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-kt * 40)) * kt) * np.exp(-kt * 25)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.005 * sr))
    for start in range(0, n - kick_len, beat):
        y[start:start + kick_len] += kick
        off = start + beat // 2
        if off + len(hat) < n:
            y[off:off + len(hat)] += 0.3 * hat
    y = gain * y / np.max(np.abs(y))
    left = y
    right = 0.9 * y + 0.05 * rng.standard_normal(n) * gain
    return np.stack([left, right], axis=1).astype(np.float32)
