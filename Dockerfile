FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but useful for pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
