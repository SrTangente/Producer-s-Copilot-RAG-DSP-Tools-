import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import resample_poly


# --- Helpers ---------------------------------------------------------------

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (normalized) for simple key estimation
_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_KRUMHANSL_MAJOR /= _KRUMHANSL_MAJOR.sum()
_KRUMHANSL_MINOR /= _KRUMHANSL_MINOR.sum()


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    # average channels (mid)
    return np.mean(y, axis=1)


def _safe_dbfs(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def _estimate_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """
    Returns (key_name, mode, confidence[0..1])
    """
    # Chromagram via CQT is more robust for key detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma = chroma.mean(axis=1)
    if chroma.sum() <= 0:
        return "C", "major", 0.0
    chroma = chroma / (np.linalg.norm(chroma) + 1e-9)

    # Correlate with rotated key profiles
    best_score = -1.0
    best_key = 0
    best_mode = "major"
    for i in range(12):
        major_prof = np.roll(_KRUMHANSL_MAJOR, i)
        minor_prof = np.roll(_KRUMHANSL_MINOR, i)

        major_prof = major_prof / (np.linalg.norm(major_prof) + 1e-9)
        minor_prof = minor_prof / (np.linalg.norm(minor_prof) + 1e-9)

        s_major = float(np.dot(chroma, major_prof))
        s_minor = float(np.dot(chroma, minor_prof))

        if s_major >= s_minor and s_major > best_score:
            best_score = s_major
            best_key = i
            best_mode = "major"
        elif s_minor > s_major and s_minor > best_score:
            best_score = s_minor
            best_key = i
            best_mode = "minor"

    # crude confidence: normalize dot product into [0,1]
    confidence = float(np.clip((best_score + 1) / 2, 0.0, 1.0))
    return KEY_NAMES[best_key], best_mode, confidence


def _analyze_loudness(y_mono: np.ndarray, sr: int) -> tuple[float, float, float]:
    """
    Returns (integrated_lufs, lra, true_peak_dbfs)
    - True peak approximated by 4x oversampling
    """
    meter = pyln.Meter(sr)  # ITU-R BS.1770 with K-weighting
    # pyloudnorm expects float32/64 in [-1, 1]
    y_float = y_mono.astype(np.float64)
    lufs_i = float(meter.integrated_loudness(y_float))
    lra = float(meter.loudness_range(y_float))

    # approximate true peak via 4x oversampling
    y_os = resample_poly(y_float, up=4, down=1)
    true_peak_dbfs = _safe_dbfs(np.max(np.abs(y_os)))

    return lufs_i, lra, true_peak_dbfs


# --- Main API --------------------------------------------------------------

def analyze_audio(path: str,
                  target_sr: int = 48000) -> dict:
    """
    Analyze an audio file and return metrics:
    {
      'sr': int,
      'duration_s': float,
      'bpm': float,
      'key': 'A',
      'mode': 'minor',
      'key_confidence': float (0..1),
      'lufs_i': float,
      'lra': float,
      'true_peak_dbfs': float,
      'spectral_centroid_mean_hz': float
    }
    """
    # Load with soundfile for speed; fall back to librosa if needed
    y, sr = sf.read(path, always_2d=False)
    # Convert to float32 in [-1, 1]
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    # Ensure mono for analysis metrics that expect mono
    y_mono = _to_mono(y)

    # Resample (consistent metrics & speed for feature extraction)
    if sr != target_sr:
        y_mono = librosa.resample(y=y_mono, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr

    duration_s = float(len(y_mono) / sr)

    # --- Tempo (BPM)
    # Disable trimming so we don’t bias tempo on silence
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr, trim=False)
    bpm = float(tempo)

    # --- Key & mode
    key_name, mode, key_conf = _estimate_key(y_mono, sr)

    # --- Loudness & true peak
    lufs_i, lra, true_peak_dbfs = _analyze_loudness(y_mono, sr)

    # --- A few extra helpful features (used for “reference deltas”)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    spectral_centroid_mean_hz = float(np.mean(spectral_centroid))

    return {
        "sr": int(sr),
        "duration_s": duration_s,
        "bpm": bpm,
        "key": key_name,
        "mode": mode,
        "key_confidence": key_conf,
        "lufs_i": lufs_i,                  # Integrated loudness (LUFS)
        "lra": lra,                        # Loudness range (LU)
        "true_peak_dbfs": true_peak_dbfs,  # Approx. dBTP via 4x oversampling
        "spectral_centroid_mean_hz": spectral_centroid_mean_hz,
    }
