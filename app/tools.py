"""DSP analysis of a mix or master: tempo, key, loudness, dynamics, tone, stereo."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, resample_poly, sosfilt, welch

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Frequency bands used for the tonal-balance profile (Hz)
BANDS = {
    "sub": (20, 60),
    "low": (60, 250),
    "low_mid": (250, 2000),
    "high_mid": (2000, 6000),
    "high": (6000, 20000),
}

FEATURE_SR = 22050          # sample rate for tempo / key / spectral features
MAX_ANALYSIS_SECONDS = 600  # ignore anything after 10 minutes


class AudioLoadError(ValueError):
    """The file could not be decoded as audio."""


# --- Loading ---------------------------------------------------------------

def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    """Decode anything ffmpeg understands (m4a, aac...) via a temporary WAV."""
    if not shutil.which("ffmpeg"):
        raise AudioLoadError("Unsupported audio format (ffmpeg is not installed).")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn",
             "-t", str(MAX_ANALYSIS_SECONDS), "-c:a", "pcm_f32le", wav],
            capture_output=True, timeout=120, check=False)
        if proc.returncode != 0 or not os.path.exists(wav):
            raise AudioLoadError("Could not decode audio: the file is not a supported audio format.")
        return sf.read(wav, always_2d=False, dtype="float32")


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (samples, sr) with samples shaped (n,) or (n, channels), float32."""
    try:
        y, sr = sf.read(path, always_2d=False, dtype="float32")
    except Exception:  # pylint: disable=broad-except
        # Formats libsndfile can't read (m4a, aac...) go through ffmpeg
        y, sr = _decode_with_ffmpeg(path)
    if y.size == 0:
        raise AudioLoadError("The file contains no audio samples.")
    y = y[: int(MAX_ANALYSIS_SECONDS * sr)]
    if y.ndim == 2 and y.shape[1] > 2:
        y = y[:, :2]  # analyse the front pair only
    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=1)


def _dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


# --- Individual analyses ---------------------------------------------------

def estimate_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """Return (key_name, mode, confidence in [0, 1])."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if chroma.sum() <= 0:
        return "C", "major", 0.0
    chroma = chroma - chroma.mean()

    scores = []
    for mode, profile in (("major", _KRUMHANSL_MAJOR), ("minor", _KRUMHANSL_MINOR)):
        for i in range(12):
            prof = np.roll(profile, i)
            prof = prof - prof.mean()
            r = float(np.dot(chroma, prof) / (np.linalg.norm(chroma) * np.linalg.norm(prof) + 1e-9))
            scores.append((r, i, mode))
    scores.sort(reverse=True)
    best_r, best_i, best_mode = scores[0]
    # Confidence: correlation strength, discounted when the runner-up is close
    margin = best_r - scores[1][0]
    confidence = float(np.clip(0.5 * (best_r + 1) * 0.7 + min(margin * 3, 0.3), 0.0, 1.0))
    return KEY_NAMES[best_i], best_mode, confidence


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    hop = 512
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    coarse = float(np.atleast_1d(tempo)[0])
    if len(beats) < 4:
        return coarse
    # The tempogram estimate is quantised; a least-squares fit of beat times
    # against beat index averages out the frame-grid jitter.
    times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    period = np.polyfit(np.arange(len(times)), times, 1)[0]
    refined = 60.0 / period if period > 0 else coarse
    # Only trust the refinement when it agrees with the coarse estimate
    return float(refined if abs(refined - coarse) / coarse < 0.08 else coarse)


def _k_weighted_block_power(y: np.ndarray, sr: int, block_s: float, hop_s: float) -> np.ndarray:
    """Mean-square power of K-weighted audio per block, summed over channels
    (BS.1770 channel weights are 1.0 for L and R)."""
    meter = pyln.Meter(sr)  # used only for its K-weighting filter coefficients
    chans = [y] if y.ndim == 1 else [y[:, c] for c in range(y.shape[1])]
    block, hop = int(round(block_s * sr)), int(round(hop_s * sr))
    n = len(chans[0])
    if n < block:
        block = n
    starts = np.arange(0, n - block + 1, hop)
    total = np.zeros(len(starts))
    for ch in chans:
        x = ch.astype(np.float64)
        for filt in meter._filters.values():  # pylint: disable=protected-access
            x = filt.apply_filter(x)
        # Sliding mean of x^2 via cumulative sums: one pass for every block
        csum = np.concatenate([[0.0], np.cumsum(x * x)])
        total += (csum[starts + block] - csum[starts]) / block
    return total


def _to_lufs(power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(power)


def integrated_loudness(y: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770-4 integrated loudness: 400 ms blocks, 75 % overlap,
    absolute gate at -70 LUFS and relative gate at -10 LU."""
    power = _k_weighted_block_power(y, sr, 0.4, 0.1)
    loud = _to_lufs(power)
    power = power[loud > -70.0]
    if power.size == 0:
        return -70.0
    rel_gate = _to_lufs(np.mean(power)) - 10.0
    power = power[_to_lufs(power) > rel_gate]
    return float(_to_lufs(np.mean(power))) if power.size else -70.0


def loudness_range(y: np.ndarray, sr: int) -> float:
    """EBU Tech 3342 loudness range (LU): spread between the 10th and 95th
    percentile of 3 s short-term loudness, gated at -70 LUFS and -20 LU."""
    if y.shape[0] < 3 * sr:
        return 0.0
    st = _to_lufs(_k_weighted_block_power(y, sr, 3.0, 1.0))
    st = st[np.isfinite(st) & (st > -70.0)]
    if st.size < 2:
        return 0.0
    rel_gate = _to_lufs(np.mean(10 ** ((st + 0.691) / 10))) - 20.0
    st = st[st > rel_gate]
    if st.size < 2:
        return 0.0
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def true_peak(y: np.ndarray, oversample: int = 4, block: int = 8192) -> float:
    """Approximate true peak (linear) with 4x polyphase oversampling.

    Only blocks whose sample peak is within 3 dB of the file's sample peak can
    hold the true peak, so only those are oversampled."""
    y2 = y.reshape(len(y), -1)
    sample_peak = float(np.max(np.abs(y2)))
    if sample_peak == 0:
        return 0.0
    n = len(y2)
    n_blocks = int(np.ceil(n / block))
    padded = np.zeros((n_blocks * block, y2.shape[1]), dtype=np.float32)
    padded[:n] = y2
    block_peaks = np.abs(padded).reshape(n_blocks, block, -1).max(axis=(1, 2))
    best = sample_peak
    margin = 64  # samples of context so the filter sees the waveform around the peak
    for b in np.nonzero(block_peaks >= sample_peak * 10 ** (-3 / 20))[0]:
        lo, hi = max(0, b * block - margin), min(n, (b + 1) * block + margin)
        seg = resample_poly(y2[lo:hi].astype(np.float64), oversample, 1, axis=0)
        # Drop the filter's edge transients: keep only the block itself
        start = (b * block - lo) * oversample
        stop = start + (min(n, (b + 1) * block) - b * block) * oversample
        best = max(best, float(np.max(np.abs(seg[start:stop]))))
    return best


def analyze_loudness(y: np.ndarray, sr: int) -> dict:
    """Integrated loudness, LRA, true peak, sample peak, PLR, clipping."""
    lufs_i = integrated_loudness(y, sr)
    lra = loudness_range(y, sr)
    sample_peak = float(np.max(np.abs(y)))
    true_peak_dbfs = _dbfs(true_peak(y))
    clipped = int(np.sum(np.abs(y) >= 0.999))
    return {
        "lufs_i": lufs_i,
        "lra": lra,
        "true_peak_dbfs": true_peak_dbfs,
        "sample_peak_dbfs": _dbfs(sample_peak),
        "plr_db": true_peak_dbfs - lufs_i,  # peak-to-loudness ratio
        "clipped_samples": clipped,
    }


def analyze_dynamics(y_mono: np.ndarray, sr: int) -> dict:
    """Crest factor and RMS spread over 400 ms windows."""
    peak = np.max(np.abs(y_mono))
    rms = np.sqrt(np.mean(y_mono ** 2))
    crest_db = _dbfs(peak) - _dbfs(rms)
    frame = int(0.4 * sr)
    if len(y_mono) >= frame * 2:
        rms_frames = librosa.feature.rms(y=y_mono, frame_length=frame, hop_length=frame // 2)[0]
        rms_db = 20 * np.log10(np.maximum(rms_frames, 1e-6))
        rms_db = rms_db[rms_db > -60]
        spread = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)) if rms_db.size else 0.0
    else:
        spread = 0.0
    return {"crest_factor_db": float(crest_db), "rms_spread_db": spread}


def analyze_tonal_balance(y_mono: np.ndarray, sr: int) -> dict:
    """Level of each frequency band relative to the whole spectrum (dB),
    plus spectral centroid and 85 % rolloff.

    Band levels are 10*log10(band power / total power): the low bands are
    near 0 dB and the top bands well below, as in any real mix. They are
    meant to be compared with a reference, not read in isolation."""
    # Welch PSD: averaged spectrum, memory-light even for long files
    freqs, power = welch(y_mono.astype(np.float64), fs=sr, nperseg=8192)
    audible = freqs >= 20
    total = power[audible].sum() + 1e-30
    bands = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, sr / 2))
        bands[name] = float(10 * np.log10(power[mask].sum() / total + 1e-12))
    # Centroid of the magnitude spectrum (matches librosa's definition)
    mag = np.sqrt(power[audible])
    centroid = float(np.sum(freqs[audible] * mag) / (mag.sum() + 1e-30))
    cumulative = np.cumsum(power[audible]) / total
    rolloff = float(freqs[audible][min(np.searchsorted(cumulative, 0.85), audible.sum() - 1)])
    return {
        "band_level_db": bands,
        "spectral_centroid_mean_hz": centroid,
        "spectral_rolloff_hz": rolloff,
    }


def _width_and_corr(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    mid, side = (left + right) / 2, (left - right) / 2
    e_mid, e_side = np.sum(mid ** 2), np.sum(side ** 2)
    width = float(e_side / (e_mid + e_side + 1e-12))
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + 1e-12
    return width, float(np.sum(left * right) / denom)


def analyze_stereo(y: np.ndarray, sr: int) -> dict:
    """Stereo width (side share of energy, 0 = mono, 1 = pure side), L/R
    correlation, and the width below 150 Hz (bass should usually be mono)."""
    if y.ndim == 1:
        return {"channels": 1, "stereo_width": 0.0, "lr_correlation": 1.0, "low_end_width": 0.0}
    left, right = y[:, 0].astype(np.float64), y[:, 1].astype(np.float64)
    width, corr = _width_and_corr(left, right)
    # Low end: decimate to ~3 kHz first (cheap), then low-pass at 150 Hz.
    # The same filter on both channels keeps their phase relationship intact.
    factor = max(1, sr // 3000)
    low = resample_poly(y.astype(np.float64), 1, factor, axis=0) if factor > 1 else y.astype(np.float64)
    sos = butter(4, 150, btype="lowpass", fs=sr / factor, output="sos")
    low = sosfilt(sos, low, axis=0)
    low_width, _ = _width_and_corr(low[:, 0], low[:, 1])
    return {"channels": 2, "stereo_width": width, "lr_correlation": corr, "low_end_width": low_width}


# --- Main API --------------------------------------------------------------

def analyze_audio(path: str) -> dict:
    """Analyze an audio file and return a flat-ish dict of metrics.

    Keys: sr, duration_s, bpm, key, mode, key_confidence, lufs_i, lra,
    true_peak_dbfs, sample_peak_dbfs, plr_db, clipped_samples, crest_factor_db,
    rms_spread_db, band_level_db{sub,low,low_mid,high_mid,high},
    spectral_centroid_mean_hz, spectral_rolloff_hz, channels, stereo_width,
    lr_correlation, low_end_width.
    """
    with warnings.catch_warnings():
        # librosa is chatty about short or tonal-less clips
        warnings.simplefilter("ignore", UserWarning)
        return _analyze(path)


def _analyze(path: str) -> dict:
    y, sr = load_audio(path)
    y_mono = _to_mono(y)
    duration_s = float(len(y_mono) / sr)

    # Loudness and stereo at the native sample rate
    metrics: dict = {"sr": sr, "duration_s": duration_s}
    metrics.update(analyze_loudness(y, sr))
    metrics.update(analyze_stereo(y, sr))

    # Musical features at ~22 kHz: integer decimation is much cheaper than
    # arbitrary-ratio resampling and keeps everything that matters below 11 kHz
    factor = max(1, int(round(sr / FEATURE_SR)))
    feat_sr = sr // factor
    y_feat = resample_poly(y_mono, 1, factor).astype(np.float32) if factor > 1 else y_mono
    metrics["bpm"] = estimate_bpm(y_feat, feat_sr)
    key, mode, conf = estimate_key(y_feat, feat_sr)
    metrics.update({"key": key, "mode": mode, "key_confidence": conf})
    metrics.update(analyze_dynamics(y_mono, sr))
    metrics.update(analyze_tonal_balance(y_mono, sr))
    return metrics
