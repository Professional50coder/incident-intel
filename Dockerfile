# Serves the trained model over FastAPI. CPU-only PyTorch on purpose: the model is a small
# conv autoencoder over 64x64 frames, inference doesn't need a GPU, and a CPU-only image is
# far smaller and portable to any host (a GPU is only useful for retraining, not serving).
FROM python:3.11-slim

# opencv-python-headless still links against a couple of shared libs at import time even
# without a GUI backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY checkpoints ./checkpoints
COPY samples ./samples
RUN pip install --no-cache-dir -e .

# Run as a non-root user with a writable data dir (Hugging Face Spaces runs containers as uid 1000).
RUN useradd -m -u 1000 app && mkdir -p /app/data && chown -R app /app
USER app
ENV INCIDENT_DB=/app/data/incident_intel.db     SAMPLES_DIR=/app/samples     PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn incident_intel.api.main:app --host 0.0.0.0 --port ${PORT}"]
