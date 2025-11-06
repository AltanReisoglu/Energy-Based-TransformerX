FROM python:3.10-slim-bullseye

# Keep Python output unbuffered and don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies needed for building Python packages and git
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml poetry.lock* /app/

# Install Poetry if pyproject exists and install dependencies. If no pyproject, skip.
RUN if [ -f pyproject.toml ]; then \
      pip install --no-cache-dir poetry && \
      poetry config virtualenvs.create false && \
      poetry install --no-dev --no-interaction --no-ansi; \
    else \
      echo "No pyproject.toml found, skipping Poetry install"; \
    fi

# Copy the rest of the project
COPY . /app

# Create a non-root user for better security (optional)
ARG UID=1000
RUN useradd -m -u $UID appuser || true
USER appuser

EXPOSE 8000

# Default command; override in compose or docker run if needed
CMD ["python", "main.py"]