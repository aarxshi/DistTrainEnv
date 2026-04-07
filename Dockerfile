FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (Docker layer caching)
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openenv-core
# copy all project files
COPY . .

# expose port for HF Spaces
EXPOSE 7860

# health check — validator pings /reset
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# run the FastAPI server
CMD ["python", "app.py"]
