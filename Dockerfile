FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and install OS dependencies if any needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main code
COPY main.py .

# We assume a data directory mapped to /app/data via volume
# Set environment variables if needed
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
