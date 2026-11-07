# JobFit API — CPU-only image (~600 MB). No GPU, no torch: embeddings run on ONNX Runtime.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    JOBFIT_MODEL_DIR=/app/models \
    JOBFIT_CACHE=/app/cache/llm_cache.sqlite

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts/download_model.py ./scripts/download_model.py
# bake the embedding model into the image so the container starts offline
RUN python scripts/download_model.py

# pre-built job index + posting texts (see scripts/prepare_data.py, scripts/build_index.py)
COPY data/processed ./data/processed

RUN useradd --create-home --uid 1000 jobfit && mkdir -p /app/cache && chown -R jobfit /app/cache /app/models
USER jobfit

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
