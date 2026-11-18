FROM python:3.12-slim

# Tuned for a small instance (512 MB, a fraction of a CPU): single-threaded
# maths and fewer malloc arenas keep memory use low and predictable.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    NUMBA_CACHE_DIR=/home/user/.numba

# libsndfile ships inside the soundfile wheel; ffmpeg decodes m4a/aac
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN mkdir -p data && chown -R user:user /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
USER user
ENV HOME=/home/user

# Do the slow first-run work now, so a cold start doesn't pay for it:
# bytecode for the app and librosa's compiled (numba) routines.
RUN python -m compileall -q app \
 && python -c "import numpy as np; from app.tools import estimate_bpm, estimate_key; y = np.random.default_rng(0).normal(0, .1, 66150).astype('float32'); estimate_bpm(y, 22050); estimate_key(y, 22050)"

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
