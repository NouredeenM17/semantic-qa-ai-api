# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Work directory
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# --- Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
# Install dependencies from wheels, then anything missed (should be minimal)
RUN pip install --no-cache /wheels/* && \
    pip install --no-cache -r requirements.txt && \
    rm -rf /wheels

# Copy application code
# Copy the application directory into the container
COPY ./app ./app
# Copy potentially other needed files like scripts if they are run inside container
# COPY ./scripts ./scripts

# Expose port 8000 for FastAPI/Uvicorn
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# Note: --reload is useful for development, remove it for production builds.
