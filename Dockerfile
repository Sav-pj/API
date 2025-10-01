# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system packages (optional but helpful for pandas wheels compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app/, models/, etc.)
COPY . .

# Expose uvicorn port
EXPOSE 8000

# Start FastAPI (ASGI app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
