"""DSP analysis of a mix or master: loudness, peaks, dynamics, tone, stereo,
tempo and key.

The file is read once, block by block, and every measurement keeps only a small
running state, so memory use doesn't grow with the length of the track. Tempo
and key, the only analyses that need the waveform in memory, use a low-rate
copy of the middle two minutes. That is what lets the app run on 512 MB.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, welch

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

BLOCK_FRAMES = 1 << 16       # frames decoded at a time
MAX_ANALYSIS_SECONDS = 600   # ignore anything after 10 minutes
FEATURE_SR = 22050           # approximate sample rate for tempo and key
EXCERPT_SECONDS = 120        # tempo and key come from the middle of the track
WELCH_SEGMENT = 8192


class AudioLoadError(ValueError):
    """The file could not be decoded as audio."""


def _dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _to_lufs(power):
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(power)


# --- Loading ---------------------------------------------------------------

@contextmanager
def _open_audio(path: str):
    """Yield an open SoundFile, decoding through ffmpeg if libsndfile can't."""
    try:
        with sf.SoundFile(path) as handle:
            yield handle
        return
    except sf.LibsndfileError:
        pass
    if not shutil.which("ffmpeg"):
        raise AudioLoadError("Unsupported audio format (ffmpeg is not installed).")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn",
             "-t", str(MAX_ANALYSIS_SECONDS), "-c:a", "pcm_f32le", wav],
            capture_output=True, timeout=300, check=False)
        if proc.returncode != 0 or not os.path.exists(wav):
            raise AudioLoadError("Could not decode audio: the file is not a supported audio format.")
        with sf.SoundFile(wav) as handle:
            yield handle


# --- Streaming meters --------------------------------------------------------

class _BinnedSum:
    """Sums a sample stream into fixed-length bins."""

    def __init__(self, bin_size: int):
        self.bin_size = bin_size
        self._carry = np.empty(0)
        self._bins: list[np.ndarray] = []

    def add(self, values: np.ndarray) -> None:
        buf = np.concatenate([self._carry, values])
        full = len(buf) // self.bin_size
        if full:
            whole = buf[:full * self.bin_size].reshape(full, self.bin_size)
            self._bins.append(whole.sum(axis=1))
        self._carry = buf[full * self.bin_size:]

    def means(self) -> np.ndarray:
        """Mean value per bin; a clip shorter than one bin yields one short bin."""
        if self._bins:
            return np.concatenate(self._bins) / self.bin_size
        return np.array([self._carry.mean()]) if self._carry.size else np.empty(0)


def _sliding_mean(values: np.ndarray, width: int, hop: int) -> np.ndarray:
    if values.size < width:
        return np.array([values.mean()]) if values.size else np.empty(0)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    return (cumulative[width:] - cumulative[:-width])[::hop] / width


def _k_weighting_sos(sr: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting: high shelf then high pass, as biquad sections."""
    gain, q, fc = 4.0, 1 / np.sqrt(2), 1500.0
    a_lin = 10 ** (gain / 40)
    w0 = 2 * np.pi * fc / sr
    alpha, cos_w0, root = np.sin(w0) / (2 * q), np.cos(w0), 2 * np.sqrt(a_lin)
    shelf = np.array([
        a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 + root * alpha),
        -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cos_w0),
        a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 - root * alpha),
        (a_lin + 1) - (a_lin - 1) * cos_w0 + root * alpha,
        2 * ((a_lin - 1) - (a_lin + 1) * cos_w0),
        (a_lin + 1) - (a_lin - 1) * cos_w0 - root * alpha,
    ])
    w0 = 2 * np.pi * 38.0 / sr
    alpha, cos_w0 = np.sin(w0) / (2 * 0.5), np.cos(w0)
    high_pass = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2,
                          1 + alpha, -2 * cos_w0, 1 - alpha])
    return np.vstack([shelf / shelf[3], high_pass / high_pass[3]])


class LoudnessMeter:
    """Integrated loudness (BS.1770-4) and loudness range (EBU Tech 3342)."""

    def __init__(self, sr: int, channels: int):
        self._sos = _k_weighting_sos(sr)
        self._state = np.zeros((2, 2, channels))
        self._bins = _BinnedSum(int(round(0.1 * sr)))

    def add(self, block: np.ndarray) -> None:
        weighted, self._state = sosfilt(self._sos, block, axis=0, zi=self._state)
        self._bins.add(np.sum(weighted * weighted, axis=1))

    def integrated(self) -> float:
        power = _sliding_mean(self._bins.means(), 4, 1)  # 400 ms blocks, 75 % overlap
        power = power[power > 0]
        power = power[_to_lufs(power) > -70.0]
        if power.size == 0:
            return -70.0
        gate = _to_lufs(np.mean(power)) - 10.0
        power = power[_to_lufs(power) > gate]
        return float(_to_lufs(np.mean(power))) if power.size else -70.0

    def loudness_range(self) -> float:
        bins = self._bins.means()
        if bins.size < 30:
            return 0.0
        power = _sliding_mean(bins, 30, 10)  # 3 s windows every second
        power = power[power > 0]
        power = power[_to_lufs(power) > -70.0]
        if power.size < 2:
            return 0.0
        short_term = _to_lufs(power)
        short_term = short_term[short_term > _to_lufs(np.mean(power)) - 20.0]
        if short_term.size < 2:
            return 0.0
        return float(np.percentile(short_term, 95) - np.percentile(short_term, 10))


class TruePeakMeter:
    """True peak by band-limited interpolation at 4x around the loudest samples.

    An inter-sample peak sits next to a local maximum of the rectified signal,
    so only those neighbourhoods need interpolating."""

    HALF_WIDTH = 8
    MARGIN_DB = 3.1  # how far below the current peak a sample can still matter

    def __init__(self, channels: int):
        h = self.HALF_WIDTH
        self._offsets = np.arange(-h + 1, h + 1)
        rows = []
        for frac in (0.25, 0.5, 0.75):
            u = frac - self._offsets
            taper = np.i0(8.0 * np.sqrt(np.clip(1 - (u / h) ** 2, 0, 1))) / np.i0(8.0)
            taps = np.sinc(u) * taper
            rows.append(taps / taps.sum())
        self._taps = np.array(rows).T
        self._tail = np.zeros((2 * h, channels))
        self.sample_peak = 0.0
        self.true_peak = 0.0

    def add(self, block: np.ndarray) -> None:
        h = self.HALF_WIDTH
        joined = np.concatenate([self._tail, block])
        self._tail = joined[-2 * h:]
        self.sample_peak = max(self.sample_peak, float(np.max(np.abs(block))))
        self.true_peak = max(self.true_peak, self.sample_peak)
        threshold = self.true_peak * 10 ** (-self.MARGIN_DB / 20)
        for c in range(joined.shape[1]):
            x = joined[:, c]
            mag = np.abs(x)
            is_peak = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:]) & (mag[1:-1] >= threshold)
            peaks = np.flatnonzero(is_peak) + 1
            # Interpolate on both sides of each peak
            bases = np.unique(np.concatenate([peaks - 1, peaks]))
            bases = bases[(bases >= h) & (bases < len(x) - h)]
            if bases.size:
                values = x[bases[:, None] + self._offsets] @ self._taps
                self.true_peak = max(self.true_peak, float(np.max(np.abs(values))))

    def finish(self) -> None:
        self.add(np.zeros((2 * self.HALF_WIDTH, self._tail.shape[1])))


class StereoMeter:
    """Width (side share of energy), L/R correlation and width below 150 Hz."""

    def __init__(self, sr: int):
        self._sos = butter(4, 150, btype="lowpass", fs=sr, output="sos")
        self._state = np.zeros((self._sos.shape[0], 2, 2))
        self._full = np.zeros(3)  # LL, RR, LR
        self._low = np.zeros(3)

    @staticmethod
    def _energies(y: np.ndarray) -> np.ndarray:
        left, right = y[:, 0], y[:, 1]
        return np.array([np.dot(left, left), np.dot(right, right), np.dot(left, right)])

    def add(self, block: np.ndarray) -> None:
        self._full += self._energies(block)
        low, self._state = sosfilt(self._sos, block, axis=0, zi=self._state)
        self._low += self._energies(low)

    @staticmethod
    def _width(energies: np.ndarray) -> float:
        ll, rr, lr = energies
        side = max((ll + rr - 2 * lr) / 4, 0.0)
        mid = max((ll + rr + 2 * lr) / 4, 0.0)
        return float(side / (mid + side + 1e-12))

    def result(self) -> dict:
        ll, rr, lr = self._full
        corr = float(lr / (np.sqrt(ll * rr) + 1e-12))
        return {"channels": 2, "stereo_width": self._width(self._full),
                "lr_correlation": corr, "low_end_width": self._width(self._low)}


class SpectrumMeter:
    """Average power spectrum (non-overlapping Welch segments) of the mono mix."""

    def __init__(self, sr: int):
        self.sr = sr
        self._carry = np.empty(0)
        self._sum = None
        self._segments = 0
        self._freqs = None

    def _add_segments(self, x: np.ndarray) -> None:
        count = len(x) // WELCH_SEGMENT
        self._freqs, power = welch(x, fs=self.sr, nperseg=WELCH_SEGMENT, noverlap=0)
        self._sum = power * count if self._sum is None else self._sum + power * count
        self._segments += count

    def add(self, mono: np.ndarray) -> None:
        buf = np.concatenate([self._carry, mono])
        usable = (len(buf) // WELCH_SEGMENT) * WELCH_SEGMENT
        if usable:
            self._add_segments(buf[:usable])
        self._carry = buf[usable:]

    def result(self) -> dict:
        if self._segments == 0:  # clip shorter than one segment
            self._add_segments(np.pad(self._carry, (0, WELCH_SEGMENT - len(self._carry))))
        freqs, power = self._freqs, self._sum / self._segments
        audible = freqs >= 20
        total = power[audible].sum() + 1e-30
        bands = {}
        for name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < min(hi, self.sr / 2))
            bands[name] = float(10 * np.log10(power[mask].sum() / total + 1e-12))
        mag = np.sqrt(power[audible])
        centroid = float(np.sum(freqs[audible] * mag) / (mag.sum() + 1e-30))
        cumulative = np.cumsum(power[audible]) / total
        rolloff = float(freqs[audible][min(np.searchsorted(cumulative, 0.85), audible.sum() - 1)])
        return {"band_level_db": bands, "spectral_centroid_mean_hz": centroid,
                "spectral_rolloff_hz": rolloff}


class ExcerptCollector:
    """Keeps a low-rate mono copy of the middle of the track."""

    def __init__(self, sr: int, total_frames: int):
        self.factor = max(1, int(round(sr / FEATURE_SR)))
        self.sr = sr / self.factor
        length = min(total_frames, EXCERPT_SECONDS * sr)
        self._start = (total_frames - length) // 2
        self._stop = self._start + length
        self._pos = 0
        self._phase = 0
        self._parts: list[np.ndarray] = []
        self._sos = None
        if self.factor > 1:
            self._sos = butter(8, 0.8 * self.sr / 2, fs=sr, output="sos")
            self._state = np.zeros((self._sos.shape[0], 2))

    def add(self, mono: np.ndarray) -> None:
        begin, self._pos = self._pos, self._pos + len(mono)
        x = mono[max(self._start - begin, 0):max(self._stop - begin, 0)]
        if x.size == 0:
            return
        if self._sos is not None:
            x, self._state = sosfilt(self._sos, x, zi=self._state)
        self._parts.append(x[self._phase::self.factor].astype(np.float32))
        self._phase = (self._phase - len(x)) % self.factor

    def result(self) -> np.ndarray:
        return np.concatenate(self._parts) if self._parts else np.empty(0, dtype=np.float32)


# --- Tempo and key -----------------------------------------------------------

def estimate_bpm(y: np.ndarray, sr: float) -> float:
    """Tempo from librosa's beat tracker, refined with a fit over the beat times."""
    import librosa  # pylint: disable=import-outside-toplevel  # heavy: load on first use

    if y.size == 0 or not np.any(y):
        return 0.0
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


def estimate_key(y: np.ndarray, sr: float) -> tuple[str, str, float]:
    """Return (key_name, mode, confidence in [0, 1])."""
    import librosa  # pylint: disable=import-outside-toplevel

    if y.size == 0 or not np.any(y):
        return "C", "major", 0.0
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


# --- Main API --------------------------------------------------------------

def loudness_range(y: np.ndarray, sr: int) -> float:
    """Loudness range (LU) of an in-memory signal."""
    samples = y.reshape(len(y), -1).astype(np.float64)
    meter = LoudnessMeter(sr, samples.shape[1])
    meter.add(samples)
    return meter.loudness_range()


def _dynamics(mono_peak: float, mono_energy: float, frames: int, rms_bins: np.ndarray) -> dict:
    rms = np.sqrt(mono_energy / max(frames, 1))
    spread = 0.0
    if rms_bins.size >= 8:
        levels = 10 * np.log10(np.maximum(_sliding_mean(rms_bins, 4, 2), 1e-12))
        levels = levels[levels > -60]
        if levels.size:
            spread = float(np.percentile(levels, 95) - np.percentile(levels, 10))
    return {"crest_factor_db": _dbfs(mono_peak) - _dbfs(rms), "rms_spread_db": spread}


def analyze_audio(path: str) -> dict:
    """Analyze an audio file and return a flat-ish dict of metrics.

    Keys: sr, duration_s, bpm, key, mode, key_confidence, lufs_i, lra,
    true_peak_dbfs, sample_peak_dbfs, plr_db, clipped_samples, crest_factor_db,
    rms_spread_db, band_level_db{sub,low,low_mid,high_mid,high},
    spectral_centroid_mean_hz, spectral_rolloff_hz, channels, stereo_width,
    lr_correlation, low_end_width.
    """
    with _open_audio(path) as audio:
        sr, channels = audio.samplerate, min(audio.channels, 2)
        limit = int(MAX_ANALYSIS_SECONDS * sr)
        total = min(audio.frames, limit)
        if total <= 0:
            raise AudioLoadError("The file contains no audio samples.")

        loudness = LoudnessMeter(sr, channels)
        peaks = TruePeakMeter(channels)
        stereo = StereoMeter(sr) if channels == 2 else None
        spectrum = SpectrumMeter(sr)
        excerpt = ExcerptCollector(sr, total)
        rms_bins = _BinnedSum(int(round(0.1 * sr)))
        frames, clipped, mono_peak, mono_energy = 0, 0, 0.0, 0.0

        while frames < limit:
            try:
                raw = audio.read(min(BLOCK_FRAMES, limit - frames), dtype="float64", always_2d=True)
            except (sf.LibsndfileError, RuntimeError) as exc:
                raise AudioLoadError(f"Could not decode audio: {exc}") from exc
            if len(raw) == 0:
                break
            block = raw[:, :channels]
            frames += len(block)
            mono = block.mean(axis=1)
            squared = mono * mono

            loudness.add(block)
            peaks.add(block)
            spectrum.add(mono)
            excerpt.add(mono)
            rms_bins.add(squared)
            if stereo:
                stereo.add(block)
            clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
            mono_peak = max(mono_peak, float(np.max(np.abs(mono))))
            mono_energy += float(squared.sum())

    if frames == 0:
        raise AudioLoadError("The file contains no audio samples.")
    peaks.finish()

    true_peak_dbfs = _dbfs(peaks.true_peak)
    lufs_i = loudness.integrated()
    metrics: dict = {
        "sr": int(sr),
        "duration_s": frames / sr,
        "lufs_i": lufs_i,
        "lra": loudness.loudness_range(),
        "true_peak_dbfs": true_peak_dbfs,
        "sample_peak_dbfs": _dbfs(peaks.sample_peak),
        "plr_db": true_peak_dbfs - lufs_i,  # peak-to-loudness ratio
        "clipped_samples": clipped,
    }
    metrics.update(stereo.result() if stereo else
                   {"channels": 1, "stereo_width": 0.0, "lr_correlation": 1.0, "low_end_width": 0.0})
    metrics.update(_dynamics(mono_peak, mono_energy, frames, rms_bins.means()))
    metrics.update(spectrum.result())

    features = excerpt.result()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # librosa warns on short clips
        metrics["bpm"] = estimate_bpm(features, excerpt.sr)
        key, mode, confidence = estimate_key(features, excerpt.sr)
    metrics.update({"key": key, "mode": mode, "key_confidence": confidence})
    return metrics
