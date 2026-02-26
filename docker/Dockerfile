FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip3 install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cu124

COPY . .
RUN mkdir -p data logs

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python3", "scripts/run_all.py"]
