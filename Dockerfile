FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# libsndfile ships inside the soundfile wheel; ffmpeg decodes m4a/aac
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN mkdir -p data && chown -R user:user /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
USER user
ENV HOME=/home/user NUMBA_CACHE_DIR=/tmp/numba

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
