import os
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tools import analyze_audio


def show_spectrogram(path: str, sr: int) -> None:
    """Display a spectrogram for the audio file."""
    y, _ = librosa.load(path, sr=sr)
    spec = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)


def show_loudness_curve(path: str, sr: int) -> None:
    """Plot a simple loudness (RMS) curve over time."""
    y, _ = librosa.load(path, sr=sr)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots()
    ax.plot(times, rms)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    ax.set_title("Loudness Curve")
    st.pyplot(fig)


def main() -> None:
    st.title("Audio Analysis Tool")
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

    if uploaded is None:
        return

    # Write the uploaded file to a temporary location for analysis
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    metrics = analyze_audio(tmp_path)
    st.subheader("Metrics")
    st.json(metrics)

    if st.checkbox("Show spectrogram"):
        show_spectrogram(tmp_path, metrics["sr"])

    if st.checkbox("Show loudness curve"):
        show_loudness_curve(tmp_path, metrics["sr"])

    os.remove(tmp_path)


if __name__ == "__main__":
    main()
