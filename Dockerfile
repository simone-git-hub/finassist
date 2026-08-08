FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINASSIST_CONFIG_PATH=/app/configs/offline.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY data ./data
COPY tests/fixtures ./tests/fixtures
COPY src ./src

RUN pip install --no-cache-dir -e .

RUN finassist-ingest --manifest data/raw/manifest.offline.csv && \
    finassist-build-index --config configs/offline.yaml

EXPOSE 8000 8501

CMD ["finassist-api", "--host", "0.0.0.0", "--port", "8000"]
