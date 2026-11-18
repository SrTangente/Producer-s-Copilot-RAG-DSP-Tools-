"""Reference tracks: per-workspace store, built-in genre targets, comparison."""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from pathlib import Path

# Metrics compared between a mix and its reference: (key, label, unit, tolerance)
# The tolerance is the difference below which the two count as "matching".
COMPARED_METRICS = [
    ("lufs_i", "Integrated loudness", "LUFS", 1.0),
    ("true_peak_dbfs", "True peak", "dBTP", 0.5),
    ("lra", "Loudness range", "LU", 2.0),
    ("plr_db", "Peak-to-loudness ratio", "dB", 1.5),
    ("stereo_width", "Stereo width", "", 0.05),
    ("spectral_centroid_mean_hz", "Brightness (spectral centroid)", "Hz", 250.0),
    ("bpm", "Tempo", "BPM", 2.0),
]
BAND_TOLERANCE_DB = 3.0
BAND_LABELS = {"sub": "sub (20-60 Hz)", "low": "low (60-250 Hz)", "low_mid": "low mids (250 Hz-2 kHz)",
               "high_mid": "upper mids (2-6 kHz)", "high": "highs (6 kHz+)"}

# Approximate, genre-typical master profiles: loudness and dynamics only.
# Tonal balance and stereo width vary too much within a genre to give a
# useful generic target; upload a reference track to compare those.
GENRE_TARGETS: dict[str, dict] = {
    "streaming": {
        "name": "Streaming-normalised master",
        "description": "Spotify / Apple Music / YouTube play back around -14 LUFS; louder masters are turned down.",
        "metrics": {"lufs_i": -14.0, "true_peak_dbfs": -1.0, "lra": 7.0, "plr_db": 13.0},
    },
    "pop": {
        "name": "Modern pop",
        "description": "Dense, bright and loud, with a controlled low end.",
        "metrics": {"lufs_i": -9.0, "true_peak_dbfs": -1.0, "lra": 5.0, "plr_db": 8.5},
    },
    "edm": {
        "name": "EDM / club",
        "description": "Very loud, heavily limited, big mono sub and a wide top end.",
        "metrics": {"lufs_i": -7.0, "true_peak_dbfs": -0.5, "lra": 4.0, "plr_db": 7.0},
    },
    "hiphop": {
        "name": "Hip-hop / trap",
        "description": "Sub-heavy 808s, upfront vocal, loud but punchy.",
        "metrics": {"lufs_i": -8.0, "true_peak_dbfs": -1.0, "lra": 5.0, "plr_db": 8.0},
    },
    "rock": {
        "name": "Rock / indie",
        "description": "Guitars fill the mids; louder than streaming but less crushed than EDM.",
        "metrics": {"lufs_i": -9.5, "true_peak_dbfs": -1.0, "lra": 6.0, "plr_db": 9.5},
    },
    "acoustic": {
        "name": "Acoustic / jazz / classical",
        "description": "Dynamics matter more than loudness.",
        "metrics": {"lufs_i": -16.0, "true_peak_dbfs": -1.0, "lra": 10.0, "plr_db": 15.0},
    },
    "lofi": {
        "name": "Lo-fi / chill",
        "description": "Warm, dark top end, moderate loudness.",
        "metrics": {"lufs_i": -12.0, "true_peak_dbfs": -1.0, "lra": 5.0, "plr_db": 10.0},
    },
}

MAX_REFERENCES_PER_WORKSPACE = 20
MAX_CLIENT_REFERENCE_CHARS = 20_000
_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def valid_workspace(workspace: str | None) -> bool:
    return bool(workspace and _WORKSPACE_RE.match(workspace))


class ReferenceStore:
    """JSON-file store of reference-track metrics, partitioned by workspace.

    Only analysis metrics are kept, never the audio itself.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def list(self, workspace: str) -> list[dict]:
        with self._lock:
            refs = self._load().get(workspace, {})
        return sorted(refs.values(), key=lambda r: r["created_at"], reverse=True)

    def get(self, workspace: str, ref_id: str) -> dict | None:
        with self._lock:
            return self._load().get(workspace, {}).get(ref_id)

    def add(self, workspace: str, name: str, metrics: dict) -> dict:
        ref = {"id": uuid.uuid4().hex[:12], "name": name[:120] or "Untitled reference",
               "created_at": time.time(), "metrics": metrics}
        with self._lock:
            data = self._load()
            refs = data.setdefault(workspace, {})
            if len(refs) >= MAX_REFERENCES_PER_WORKSPACE:
                oldest = min(refs.values(), key=lambda r: r["created_at"])
                refs.pop(oldest["id"])
            refs[ref["id"]] = ref
            self._save(data)
        return ref

    def delete(self, workspace: str, ref_id: str) -> bool:
        with self._lock:
            data = self._load()
            removed = data.get(workspace, {}).pop(ref_id, None)
            if removed:
                self._save(data)
        return removed is not None


def parse_client_reference(raw: str) -> dict:
    """Validate reference metrics sent by the browser, keeping known numbers only."""
    if len(raw) > MAX_CLIENT_REFERENCE_CHARS:
        raise ValueError("Reference data is too large")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Reference data must be an object")

    def number(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    clean: dict = {}
    for key, *_ in COMPARED_METRICS:
        value = number(data.get(key))
        if value is not None:
            clean[key] = value
    bands = data.get("band_level_db")
    if isinstance(bands, dict):
        levels = {band: number(bands.get(band)) for band in BAND_LABELS}
        clean["band_level_db"] = {k: v for k, v in levels.items() if v is not None}
    if not clean:
        raise ValueError("Reference data has no usable metrics")
    return clean


def clean_reference_name(name: str | None) -> str:
    return re.sub(r"[<>\"\x00-\x1f]", "", name or "").strip()[:120] or "your reference"


def compare(metrics: dict, reference: dict) -> list[dict]:
    """Per-metric deltas (mix minus reference) for every metric the reference has."""
    rows = []
    for key, label, unit, tol in COMPARED_METRICS:
        if key in reference and key in metrics:
            delta = float(metrics[key]) - float(reference[key])
            rows.append({"metric": key, "label": label, "unit": unit,
                         "mix": float(metrics[key]), "reference": float(reference[key]),
                         "delta": delta, "within_tolerance": abs(delta) <= tol})
    ref_bands = reference.get("band_level_db")
    mix_bands = metrics.get("band_level_db")
    if ref_bands and mix_bands:
        for band, ref_val in ref_bands.items():
            if band not in mix_bands:
                continue
            delta = float(mix_bands[band]) - float(ref_val)
            rows.append({"metric": f"band_level_db.{band}", "label": f"Band level: {BAND_LABELS.get(band, band)}",
                         "unit": "dB", "mix": float(mix_bands[band]), "reference": float(ref_val),
                         "delta": delta, "within_tolerance": abs(delta) <= BAND_TOLERANCE_DB})
    return rows
